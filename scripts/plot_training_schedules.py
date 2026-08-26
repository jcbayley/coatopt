#!/usr/bin/env python3
"""Plot constraint-threshold and LR/entropy schedules from a training run.

Shows how the interleaved constraint schedule behaves during hppo_sequential
training: constraint caps step up in blocks (one level per sweep of the
objectives), while the actual thresholds are sampled uniformly within the
current cap each episode (randomise_constraints=true). Also plots the cosine
LR and entropy-coefficient schedules, including warm restarts.

Accepts either kind of training output:
  - a run directory containing training_history.csv (written by
    coatopt.utils.training_plots.save_training_curves), or the CSV itself
  - an MLflow run directory (contains metrics/), or any directory above one
    (e.g. an mlruns/ root) — the run with the longest schedule.lr history
    is chosen

Usage:
  python scripts/plot_training_schedules.py <run_path> [-o out.png]
"""

import argparse
import configparser
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same series colours as coatopt.utils.training_plots
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_WARMUP_SHADE = "#efeee9"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_mlflow_run(run_dir: Path) -> pd.DataFrame:
    """Read every metric file of an MLflow run into an episode-indexed frame.

    Metric files are lines of "<timestamp> <value> <step>"; step is the
    episode number the trainer logged at.
    """
    frames = []
    for metric_file in sorted((run_dir / "metrics").iterdir()):
        if not metric_file.is_file():
            continue
        rows = []
        for line in metric_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                rows.append((int(parts[2]), float(parts[1])))
        if rows:
            frames.append(
                pd.DataFrame(rows, columns=["episode", metric_file.name])
                .drop_duplicates("episode", keep="last")
                .set_index("episode")
            )
    if not frames:
        raise SystemExit(f"No metrics found in {run_dir / 'metrics'}")
    return pd.concat(frames, axis=1).sort_index().reset_index()


def _find_mlflow_run(path: Path) -> Optional[Path]:
    """Return the MLflow run dir at/under path with the longest lr history."""
    if (path / "metrics").is_dir():
        return path
    candidates = [
        (sum(1 for _ in f.open()), f.parent.parent)
        for f in path.glob("**/metrics/schedule.lr")
    ]
    if not candidates:
        return None
    n_lines, run_dir = max(candidates)
    print(f"Using MLflow run {run_dir} ({n_lines} logged points)")
    return run_dir


def _read_params(run_dir: Path) -> Dict[str, str]:
    """Schedule parameters from config.ini (run output dir) or params/ (MLflow)."""
    params: Dict[str, str] = {}
    config_path = run_dir / "config.ini"
    if config_path.is_file():
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        for section in ("hppo_sequential", "general"):
            if cfg.has_section(section):
                params.update(cfg[section])
        if cfg.has_option("data", "optimise_parameters"):
            params["optimise_parameters"] = cfg.get("data", "optimise_parameters")
    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for f in params_dir.iterdir():
            if f.is_file():
                params[f.name] = f.read_text().strip()
    return params


def load_run(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return (history frame with an 'episode' column, schedule params)."""
    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path), _read_params(path.parent)
    if not path.is_dir():
        raise SystemExit(f"{path} is not a run directory or CSV")
    csv = path / "training_history.csv"
    if csv.is_file():
        return pd.read_csv(csv), _read_params(path)
    mlflow_run = _find_mlflow_run(path)
    if mlflow_run is not None:
        return _load_mlflow_run(mlflow_run), _read_params(mlflow_run)
    raise SystemExit(
        f"No training_history.csv or MLflow metrics found under {path}"
    )


# ---------------------------------------------------------------------------
# Schedule reconstruction
# ---------------------------------------------------------------------------


def constraint_cap_fraction(
    episodes: np.ndarray,
    warmup_episodes: int,
    episodes_per_step: int,
    steps_per_objective: int,
    n_objectives: int,
) -> np.ndarray:
    """Fraction of warmup-best reward the constraint cap sits at, per episode.

    Mirrors the trainer: level steps up after each full sweep of the
    objectives and wraps around after steps_per_objective levels.
    """
    constrained = episodes - warmup_episodes
    phase = (constrained - 1) // episodes_per_step
    level = (phase // n_objectives) % steps_per_objective
    frac = (level + 1) / steps_per_objective
    return np.where(constrained > 0, frac, 0.0)


def _get_int(params: Dict[str, str], key: str) -> Optional[int]:
    try:
        return int(float(params[key]))
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style(ax, title: str, ylabel: str = "", logy: bool = False) -> None:
    ax.set_title(title, fontsize=10, color=_INK, loc="left")
    ax.grid(color=_GRID, linewidth=0.7)
    ax.tick_params(colors=_MUTED, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=_MUTED)
    if logy:
        ax.set_yscale("log")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _shade_schedule_structure(
    ax, params: Dict[str, str], n_objectives: int, ep_max: float
) -> None:
    """Shade the warmup region and mark constraint-block boundaries."""
    warmup = _get_int(params, "warmup_episodes")
    if warmup:
        ax.axvspan(0, min(warmup, ep_max), color=_WARMUP_SHADE, zorder=0)
    eps_per_step = _get_int(params, "episodes_per_step")
    if warmup and eps_per_step and n_objectives:
        block = eps_per_step * n_objectives  # one sweep of all objectives
        for edge in np.arange(warmup, ep_max, block):
            ax.axvline(edge, color=_GRID, lw=0.7, zorder=0)


def plot_schedules(
    df: pd.DataFrame, params: Dict[str, str], out_path: Path
) -> None:
    df = df.sort_values("episode")
    ep = df["episode"].values
    objectives: List[str] = [
        c.split(".", 1)[1] for c in df.columns if c.startswith("constraint.")
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
    )
    ax_con, ax_ent, ax_lr = axes
    for ax in axes:
        _shade_schedule_structure(ax, params, len(objectives), ep.max())

    # --- Panel 1: constraint thresholds -----------------------------------
    for i, obj in enumerate(objectives):
        color = _SERIES[i % len(_SERIES)]
        vals = df[f"constraint.{obj}"].values.astype(float)
        active = vals > 0  # 0 = warmup, or this objective is the target
        ax_con.scatter(
            ep[active],
            vals[active],
            s=9,
            color=color,
            alpha=0.65,
            lw=0,
            label=obj,
        )

    # Exact block-cap staircase: (level+1)/steps_per_objective * warmup best
    warmup = _get_int(params, "warmup_episodes")
    eps_per_step = _get_int(params, "episodes_per_step")
    steps_per_obj = _get_int(params, "steps_per_objective")
    have_schedule = all(v is not None for v in (warmup, eps_per_step, steps_per_obj))
    if have_schedule:
        grid = np.arange(ep.min(), ep.max() + 1)
        frac = constraint_cap_fraction(
            grid, warmup, eps_per_step, steps_per_obj, len(objectives)
        )
        frac = np.where(frac > 0, frac, np.nan)  # hide the cap during warmup
        for i, obj in enumerate(objectives):
            best_col = f"warmup_best.{obj}"
            if best_col not in df:
                continue
            best = df[best_col].dropna()
            if best.empty:
                continue
            ax_con.plot(
                grid,
                frac * float(best.iloc[-1]),
                color=_SERIES[i % len(_SERIES)],
                lw=1.0,
                ls="--",
                alpha=0.8,
            )
    ax_con.legend(
        fontsize=8,
        frameon=False,
        title="objective",
        title_fontsize=8,
        loc="upper left",
        markerscale=2.0,
    )
    subtitle = "dots = sampled thresholds (uniform within block cap)"
    if have_schedule:
        subtitle += ", dashed = block cap"
    _style(
        ax_con,
        f"Constraint thresholds — {subtitle}",
        ylabel="threshold (reward units)",
    )
    if warmup:
        ax_con.text(
            warmup / 2,
            0.5,
            "warmup\n(no constraints)",
            transform=ax_con.get_xaxis_transform(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
            color=_MUTED,
        )

    # --- Panels 2 & 3: entropy and LR schedules ---------------------------
    if "schedule.ent_coef" in df:
        ax_ent.plot(ep, df["schedule.ent_coef"].values, color=_SERIES[3], lw=1.4)
    _style(ax_ent, "Entropy coefficient schedule", ylabel="ent_coef")

    if "schedule.lr" in df:
        ax_lr.plot(ep, df["schedule.lr"].values, color=_SERIES[0], lw=1.4)
    _style(ax_lr, "Learning-rate schedule", ylabel="lr", logy=True)
    ax_lr.set_xlabel("episode", fontsize=9, color=_MUTED)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="run output dir / training_history.csv / MLflow run dir or mlruns root",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: <run_path>/training_schedules.png)",
    )
    args = parser.parse_args()

    df, params = load_run(args.run_path)
    if "episode" not in df.columns:
        raise SystemExit("Loaded history has no 'episode' column")
    base = args.run_path if args.run_path.is_dir() else args.run_path.parent
    out = args.out or base / "training_schedules.png"
    plot_schedules(df, params, out)


if __name__ == "__main__":
    main()
