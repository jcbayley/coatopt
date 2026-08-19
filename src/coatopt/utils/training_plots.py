#!/usr/bin/env python3
"""Training-curve plots for HPPO runs.

Renders a single PNG (training_curves.png) plus a CSV (training_history.csv)
from the metrics history collected during training, so run health is visible
without digging into MLflow.
"""

from pathlib import Path
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
]
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"


def _style(ax, title, ylabel="", logy=False):
    ax.set_title(title, fontsize=10, color=_INK)
    ax.grid(color=_GRID, linewidth=0.7)
    ax.tick_params(colors=_MUTED, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=_MUTED)
    if logy:
        ax.set_yscale("log")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_training_curves(
    history: Union[List[Dict], pd.DataFrame],
    save_dir: Union[str, Path],
    objectives: List[str],
) -> None:
    """Write training_history.csv and training_curves.png into save_dir.

    Tolerant of missing columns: each panel plots whatever is available.
    """
    save_dir = Path(save_dir)
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    if df.empty or "episode" not in df.columns:
        return
    df.to_csv(save_dir / "training_history.csv", index=False)

    ep = df["episode"].values
    n_obj_panels = len(objectives)
    n_cols = 3
    # 6 fixed panels + constraint thresholds + one per objective
    n_rows = 2 + int(np.ceil((1 + n_obj_panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.1 * n_rows), facecolor="white"
    )
    axes = axes.flatten()

    # 1: episode reward
    ax = axes[0]
    if "episode.reward_mean" in df:
        m = df["episode.reward_mean"].values
        ax.plot(ep, m, color=_SERIES[0], lw=1.5)
        if "episode.reward_std" in df:
            s = df["episode.reward_std"].values
            ax.fill_between(ep, m - s, m + s, color=_SERIES[0], alpha=0.15, lw=0)
    _style(ax, "Episode reward (100-ep mean ± std)")

    # 2: hypervolume + pareto size
    ax = axes[1]
    if "pareto.hypervolume" in df:
        ax.plot(ep, df["pareto.hypervolume"].values, color=_SERIES[1], lw=1.5)
    _style(ax, "Reward-space hypervolume")

    # 3: episode length — short stacks signal early air termination
    ax = axes[2]
    if "episode.length_mean" in df:
        m = df["episode.length_mean"].values
        ax.plot(ep, m, color=_SERIES[2], lw=1.5)
        if "episode.length_std" in df:
            s = df["episode.length_std"].values
            ax.fill_between(ep, m - s, m + s, color=_SERIES[2], alpha=0.15, lw=0)
        if "episode.length_max" in df:
            ax.plot(
                ep,
                df["episode.length_max"].values,
                color=_MUTED,
                lw=0.8,
                ls="--",
                label="max",
            )
            ax.legend(fontsize=7, frameon=False)
    _style(ax, "Episode length (layers placed)")

    # 4: policy-side losses
    ax = axes[3]
    for i, key in enumerate(["ppo.policy_loss", "ppo.entropy", "ppo.bc_loss"]):
        if key in df:
            ax.plot(
                ep,
                df[key].values,
                color=_SERIES[i % len(_SERIES)],
                lw=1.0,
                label=key.split(".")[1],
            )
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Policy losses")

    # 5: value loss
    ax = axes[4]
    if "ppo.value_loss" in df:
        v = df["ppo.value_loss"].values
        ax.plot(ep, v, color=_SERIES[3], lw=1.2)
        if np.all(v[np.isfinite(v)] > 0):
            ax.set_yscale("log")
    _style(ax, "Value loss")

    # 6: schedules
    ax = axes[5]
    for i, key in enumerate(["schedule.lr", "schedule.ent_coef"]):
        if key in df:
            ax.plot(
                ep,
                df[key].values,
                color=_SERIES[i % len(_SERIES)],
                lw=1.2,
                label=key.split(".")[1],
            )
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "LR / entropy schedule", logy=True)

    # 7: constraint thresholds
    ax = axes[6]
    for i, obj in enumerate(objectives):
        col = f"constraint.{obj}"
        if col in df:
            ax.plot(
                ep, df[col].values, color=_SERIES[i % len(_SERIES)], lw=1.0, label=obj
            )
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Constraint thresholds (0 = warmup)")

    # 8+: per-objective best values (rolling window best)
    for i, obj in enumerate(objectives):
        ax = axes[7 + i]
        col = f"vals.{obj}_best"
        if col in df:
            v = df[col].values
            ax.plot(ep, v, color=_SERIES[i % len(_SERIES)], lw=1.2)
            ok = np.isfinite(v) & (v != 0)
            if ok.any() and np.all(v[ok] > 0):
                ax.set_yscale("log")
        _style(ax, f"{obj} (best in 100-ep window)")
        ax.set_xlabel("episode", fontsize=8, color=_MUTED)

    # Hide unused axes
    for j in range(7 + n_obj_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Training curves", fontsize=13, color=_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_dir / "training_curves.png", dpi=130)
    plt.close(fig)
