#!/usr/bin/env python3
"""
Design-space diversity projection for a Pareto front.

Projects the high-dimensional coating designs (thicknesses + materials) to 2D
using t-SNE or PCA, with each subplot coloured by one objective value.
Reveals families of similar designs within the Pareto set.

Encoding
--------
  Thicknesses  → z-score normalised
  Materials    → one-hot encoded (integer indices carry no distance meaning)

Run with:
    uv run python -m coatopt.utils.plot_design_diversity --run-dir <path>
    uv run python -m coatopt.utils.plot_design_diversity --run-dir <path> --method pca
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cluster colours (tab10 palette)
_CLUSTER_COLORS = plt.cm.tab10.colors


def _build_features(designs_df: pd.DataFrame) -> np.ndarray:
    """Encode designs as a numeric matrix suitable for distance-based methods."""
    from sklearn.preprocessing import StandardScaler

    thick_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("thickness_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )
    mat_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("material_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )

    T = StandardScaler().fit_transform(designs_df[thick_cols].values.astype(float))

    mat_vals = designs_df[mat_cols].values.astype(int)
    n_mat = int(mat_vals.max()) + 1
    n = len(designs_df)
    M = np.zeros((n, len(mat_cols) * n_mat), dtype=float)
    for ci in range(mat_vals.shape[1]):
        for ri in range(n):
            M[ri, ci * n_mat + mat_vals[ri, ci]] = 1.0

    return np.hstack([M])


def _filter_by_length(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    min_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove designs whose length (index of first air layer, material==0) is below min_length."""
    mat_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("material_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )
    mat_vals = designs_df[mat_cols].values.astype(int)
    air_positions = np.array(
        [
            int(np.where(row == 0)[0][0]) if (row == 0).any() else mat_vals.shape[1]
            for row in mat_vals
        ]
    )
    mask = air_positions >= min_length
    n_removed = (~mask).sum()
    print(f"  Length filter (min={min_length}): removed {n_removed}, kept {mask.sum()}")
    return designs_df[mask].reset_index(drop=True), values_df[mask].reset_index(
        drop=True
    )


def _compute_embedding(
    X: np.ndarray,
    method: str,
    perplexity: float,
    seed: int,
    n: int,
) -> tuple[np.ndarray, str]:
    """Run t-SNE or PCA and return (embedding, label)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == "tsne":
        p = min(perplexity, max(2.0, n / 3.0))
        emb = TSNE(
            n_components=2,
            perplexity=p,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=2000,
        ).fit_transform(X)
        label = f"t-SNE (perplexity={p:.0f})"
    else:
        emb = PCA(n_components=2, random_state=seed).fit_transform(X)
        label = "PCA"
    return emb, label


def plot_design_diversity(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    save_dir: Path,
    method: str = "tsne",
    perplexity: float = 30.0,
    seed: int = 42,
    fmt: str = "png",
    min_length: int = 10,
):
    """Project Pareto designs to 2D and colour each point by objective value.

    Args:
        designs_df: DataFrame with thickness_* and material_* columns.
        values_df:  DataFrame with one column per objective (physical values).
        save_dir:   Directory to write design_diversity_<method>.<fmt>.
        method:     'tsne' or 'pca'.
        perplexity: t-SNE perplexity (ignored for PCA).
        seed:       Random seed.
        fmt:        Output format ('png', 'pdf', 'svg').
        min_length: Minimum design length (index of first air layer). Designs
                    shorter than this are excluded before projection.
    """
    try:
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.manifold import TSNE  # noqa: F401
    except ImportError:
        print("scikit-learn not found — skipping design diversity plot.")
        return

    save_dir = Path(save_dir)
    n = len(designs_df)
    if n < 4 or designs_df.empty:
        print(f"  Too few designs ({n}) for diversity projection — skipping.")
        return

    print(f"  Plotting design diversity [{method.upper()}] …")
    if min_length > 0:
        designs_df, values_df = _filter_by_length(designs_df, values_df, min_length)
        n = len(designs_df)
        if n < 4 or designs_df.empty:
            print(f"  Too few designs ({n}) after length filter — skipping.")
            return

    X = _build_features(designs_df)
    emb, method_label = _compute_embedding(X, method, perplexity, seed, n)

    obj_names = list(values_df.columns)
    ncols = min(len(obj_names), 3)
    nrows = int(np.ceil(len(obj_names) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols + 0.6, 3.2 * nrows), squeeze=False
    )
    for idx in range(len(obj_names), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    for idx, obj in enumerate(obj_names):
        ax = axes[idx // ncols, idx % ncols]
        c_vals = values_df[obj].values.astype(float)
        lbl = obj.replace("_", " ").title()

        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=c_vals,
            cmap="viridis",
            norm=mcolors.Normalize(vmin=c_vals.min(), vmax=c_vals.max()),
            s=20,
            alpha=0.85,
            linewidths=0.3,
            edgecolors="k",
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(lbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel(f"{method_label.split()[0]} 1", fontsize=8)
        ax.set_ylabel(f"{method_label.split()[0]} 2", fontsize=8)
        ax.set_title(f"Colour: {lbl}", fontsize=8)
        ax.tick_params(direction="in", top=True, right=True, labelsize=7)

    fig.suptitle(
        f"Design-space diversity · {n} Pareto solutions · {method_label}",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()

    out = save_dir / f"design_diversity_{method}.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=150 if fmt == "png" else None)
    print(f"  Saved {out.name}")
    plt.close(fig)


def _hdbscan_labels(emb: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Cluster with HDBSCAN. Noise points (label=-1) are collected into a
    final 'sparse' cluster so nothing is silently discarded."""
    from sklearn.cluster import HDBSCAN

    raw = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb)

    # Re-label noise (-1) as a new cluster id so they appear in the scatter
    if (raw == -1).any():
        noise_id = raw.max() + 1
        raw = raw.copy()
        raw[raw == -1] = noise_id

    return raw


def plot_cluster_designs(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    save_dir: Path,
    materials: dict,
    method: str = "tsne",
    perplexity: float = 30.0,
    seed: int = 42,
    max_clusters: int = 8,
    n_clusters: int = None,
    min_cluster_size: int = 3,
    n_examples: int = 2,
    fmt: str = "png",
    min_length: int = 0,
):
    """Cluster Pareto designs in embedding space and plot representative stacks.

    Auto mode (default): uses HDBSCAN, which finds tight clusters of varying
    density without assuming equal sizes. Tune min_cluster_size to control
    how many points must be together to form a cluster.

    Manual mode (n_clusters set): uses KMeans with a fixed k.

    Produces one figure:
      - Top panel: 2D scatter coloured by cluster
      - Bottom panels: n_examples coating stacks per cluster (closest to centroid)

    Args:
        designs_df:        DataFrame with thickness_* and material_* columns.
        values_df:         DataFrame with one column per objective (physical values).
        save_dir:          Directory to write design_clusters_<method>.<fmt>.
        materials:         Material properties dict.
        method:            'tsne' or 'pca'.
        perplexity:        t-SNE perplexity (ignored for PCA).
        seed:              Random seed.
        max_clusters:      Unused in auto mode (kept for CLI compat).
        n_clusters:        If set, bypass HDBSCAN and use KMeans with this k.
        min_cluster_size:  HDBSCAN min points per cluster (auto mode only).
        n_examples:        Representative designs shown per cluster.
        fmt:               Output format.
        min_length:        Minimum design length (index of first air layer). Designs
                           shorter than this are excluded before clustering.
    """
    try:
        from sklearn.cluster import HDBSCAN, KMeans  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.manifold import TSNE  # noqa: F401
    except ImportError:
        print("scikit-learn not found — skipping cluster design plot.")
        return

    from coatopt.utils.plotting import plot_coating_stack

    save_dir = Path(save_dir)
    n = len(designs_df)
    if n < 4 or designs_df.empty:
        print(f"  Too few designs ({n}) for cluster plot — skipping.")
        return

    if min_length > 0:
        designs_df, values_df = _filter_by_length(designs_df, values_df, min_length)
        n = len(designs_df)
        if n < 4 or designs_df.empty:
            print(f"  Too few designs ({n}) after length filter — skipping.")
            return

    X = _build_features(designs_df)
    emb, method_label = _compute_embedding(X, method, perplexity, seed, n)

    if n_clusters is not None:
        k = min(n_clusters, n // 2)
        print(f"  Plotting design clusters [{method.upper()}, k={k} (manual KMeans)] …")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(emb)
        centroids = km.cluster_centers_
    else:
        print(
            f"  Plotting design clusters [{method.upper()}, HDBSCAN min_size={min_cluster_size}] …"
        )
        labels = _hdbscan_labels(emb, min_cluster_size)
        k = labels.max() + 1
        # Compute centroids as mean of each cluster
        centroids = np.array([emb[labels == ci].mean(axis=0) for ci in range(k)])
        print(f"    Found k={k} clusters")

    # Find n_examples representatives per cluster (closest to centroid)
    thick_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("thickness_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )
    mat_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("material_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )
    obj_names = list(values_df.columns)

    cluster_reps = []  # list of (cluster_idx, [global_idx, ...])
    for ci in range(k):
        mask = labels == ci
        if not mask.any():
            continue
        global_idxs = np.where(mask)[0]
        dists = np.linalg.norm(emb[global_idxs] - centroids[ci], axis=1)
        top = global_idxs[np.argsort(dists)[:n_examples]]
        cluster_reps.append((ci, list(top)))

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Rows: 1 scatter + n_examples design rows; cols: one per cluster
    n_cols = len(cluster_reps)
    n_design_rows = n_examples
    height_ratios = [1.2] + [1.0] * n_design_rows
    fig = plt.figure(figsize=(max(8, 2.8 * n_cols), 4 + 3.5 * n_design_rows))
    gs = gridspec.GridSpec(
        1 + n_design_rows,
        n_cols,
        height_ratios=height_ratios,
        hspace=0.5,
        wspace=0.35,
    )

    # ── Scatter row ───────────────────────────────────────────────────────────
    ax_scatter = fig.add_subplot(gs[0, :])
    for ci in range(k):
        mask = labels == ci
        color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        ax_scatter.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=[color],
            s=25,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            label=f"Cluster {ci + 1}",
        )
        ax_scatter.scatter(
            centroids[ci, 0],
            centroids[ci, 1],
            c=[color],
            s=120,
            marker="*",
            edgecolors="k",
            linewidths=0.6,
            zorder=5,
        )

    # Ring all representative points
    for ci, global_idxs in cluster_reps:
        color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        for gi in global_idxs:
            ax_scatter.scatter(
                emb[gi, 0],
                emb[gi, 1],
                facecolors="none",
                edgecolors=color,
                s=80,
                linewidths=1.5,
                zorder=6,
            )

    ax_scatter.set_xlabel(f"{method_label.split()[0]} 1", fontsize=9)
    ax_scatter.set_ylabel(f"{method_label.split()[0]} 2", fontsize=9)
    ax_scatter.set_title(
        f"Design clusters · {n} Pareto solutions · {method_label} · k={k}",
        fontsize=9,
    )
    ax_scatter.legend(fontsize=7, markerscale=1.2, loc="best")
    ax_scatter.tick_params(direction="in", top=True, right=True, labelsize=7)

    # ── Design rows ───────────────────────────────────────────────────────────
    for col, (ci, global_idxs) in enumerate(cluster_reps):
        color = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        for row_i, gi in enumerate(global_idxs):
            ax = fig.add_subplot(gs[1 + row_i, col])

            row = designs_df.iloc[gi]
            thicknesses = row[thick_cols].values.astype(float)
            mat_indices = row[mat_cols].values.astype(int)
            active = thicknesses > 1e-12
            plot_coating_stack(
                thicknesses=thicknesses[active],
                material_indices=mat_indices[active],
                materials=materials,
                ax=ax,
                convert_to_nm=True,
            )

            obj_strs = []
            for obj in obj_names:
                val = values_df.iloc[gi][obj]
                obj_strs.append(
                    f"{obj[:4]}={val:.2e}"
                    if (abs(val) < 1e-3 or abs(val) > 1e4)
                    else f"{obj[:4]}={val:.3f}"
                )
            label = f"C{ci + 1} ex{row_i + 1}" if row_i > 0 else f"Cluster {ci + 1}"
            ax.set_title(
                f"{label}\n" + "  ".join(obj_strs),
                fontsize=7,
                color=color,
                fontweight="bold",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    out = save_dir / f"design_clusters_{method}.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=150 if fmt == "png" else None)
    print(f"  Saved {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    from coatopt.utils.utils import load_materials, load_pareto_front

    ap = argparse.ArgumentParser(
        description="Design-space diversity projection for a Pareto front.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--run-dir", required=True, help="Run directory containing pareto_front.csv"
    )
    ap.add_argument("--materials", required=True, help="Path to materials.json")
    ap.add_argument(
        "--method",
        choices=["tsne", "pca"],
        default="tsne",
        help="Projection method (default: tsne)",
    )
    ap.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30; ignored for PCA)",
    )
    ap.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fix number of clusters with KMeans, bypassing HDBSCAN auto-detection",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN min points per cluster (auto mode only, default: 3)",
    )
    ap.add_argument(
        "--n-examples",
        type=int,
        default=2,
        help="Representative designs shown per cluster (default: 2)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--fmt",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    mats = load_materials(args.materials)
    designs_df, values_df, _ = load_pareto_front(run_dir)

    min_length = 18
    plot_design_diversity(
        designs_df,
        values_df,
        run_dir,
        method=args.method,
        perplexity=args.perplexity,
        seed=args.seed,
        fmt=args.fmt,
        min_length=min_length,
    )
    plot_cluster_designs(
        designs_df,
        values_df,
        run_dir,
        materials=mats,
        method=args.method,
        perplexity=args.perplexity,
        seed=args.seed,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        n_examples=args.n_examples,
        fmt=args.fmt,
        min_length=min_length,
    )
