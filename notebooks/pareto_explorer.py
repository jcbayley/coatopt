import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import altair as alt
    from pathlib import Path
    from wigglystuff import ParallelCoordinates

    from coatopt.utils.utils import load_pareto_front, load_materials
    from coatopt.utils.plotting import plot_coating_stack

    return (
        ParallelCoordinates,
        Path,
        alt,
        load_materials,
        load_pareto_front,
        mo,
        np,
        pd,
        plot_coating_stack,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pareto Front Explorer

    **Step 1** — select a `pareto_front.csv` (browse into the run directory).
    **Step 2** — click *Compute embedding* to project the design space.
    **Step 3** — brush-select the scatter to inspect coating stacks.
    """)
    return


@app.cell
def _(Path, mo):
    _initial = str(
        next(
            (
                p.absolute()
                for p in [Path("../experiments"), Path("experiments"), Path(".")]
                if p.exists()
            ),
            Path(".").absolute(),
        )
    )
    file_browser = mo.ui.file_browser(
        initial_path=_initial,
        filetypes=[".csv"],
        multiple=False,
        label="Select pareto_front.csv",
    )
    embedding_method = mo.ui.dropdown(
        options={"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"},
        value="PCA",
        label="Embedding method",
    )
    mo.hstack([file_browser, embedding_method], gap="2rem", align="start")
    return embedding_method, file_browser


@app.cell
def _(Path, file_browser, load_materials, load_pareto_front, mo):
    _val = file_browser.value
    _csv = Path(_val[0].path) if _val else None
    _run_dir = _csv.parent if _csv else None

    designs_df = None
    values_df = None
    materials = {}

    if _run_dir and (_run_dir / "pareto_front.csv").exists():
        try:
            _designs, _values, _ = load_pareto_front(_run_dir)
            # Deduplicate on design columns
            _mask = ~_designs.duplicated()
            designs_df = _designs[_mask].reset_index(drop=True)
            values_df = _values[_mask].reset_index(drop=True)
        except Exception as _e:
            designs_df = None
            values_df = None

        for _mat_path in [
            _run_dir / "materials.json",
            _run_dir.parent / "materials.json",
            _run_dir.parent.parent / "materials.json",
        ]:
            if _mat_path.exists():
                materials = load_materials(str(_mat_path))
                break

    if designs_df is not None:
        _n = len(designs_df)
        _objs = list(values_df.columns)
        _mat_str = (
            ", ".join(v["name"] for k, v in materials.items() if k > 0)
            if materials
            else "—"
        )
        _thick_cols = [
            c
            for c in designs_df.columns
            if c.startswith("thickness_") and c.split("_")[1].isdigit()
        ]
        _n_layers = len(_thick_cols)
        mo.callout(
            mo.md(
                f"**{_n} designs** &nbsp;·&nbsp; {_n_layers} layers &nbsp;·&nbsp; "
                f"objectives: `{'`, `'.join(_objs)}` &nbsp;·&nbsp; materials: {_mat_str}"
            ),
            kind="success",
        )
    elif _run_dir and not (_run_dir / "pareto_front.csv").exists():
        mo.callout(
            mo.md(
                f"No `pareto_front.csv` found in `{_run_dir.name}` — navigate into a run subdirectory."
            ),
            kind="warn",
        )
    else:
        mo.callout(
            mo.md("Select a `pareto_front.csv` file using the browser above."),
            kind="warn",
        )
    return designs_df, materials, values_df


@app.cell
def _(mo, values_df):
    _opts = list(values_df.columns) if values_df is not None else ["(none)"]
    color_by = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Color by",
    )
    color_by
    return (color_by,)


@app.cell
def _(mo):
    compute_btn = mo.ui.run_button(label="⚡  Compute embedding")
    compute_btn
    return (compute_btn,)


@app.cell
def _(compute_btn, designs_df, embedding_method, mo, np, pd, values_df):
    mo.stop(
        not compute_btn.value,
        mo.callout(
            mo.md(
                "Click **Compute embedding** above — PCA is instant, t-SNE and UMAP take a few seconds."
            ),
            kind="info",
        ),
    )
    mo.stop(
        designs_df is None,
        mo.callout(mo.md("Load a run directory first."), kind="warn"),
    )

    _method = embedding_method.value

    if _method == "umap":
        try:
            from umap import UMAP as _UMAP
            from sklearn.preprocessing import StandardScaler as _SS

            _X = _build_features(designs_df)
            _X = _SS().fit_transform(_X)
            _coords = _UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            ).fit_transform(_X)
            _label = "UMAP"
        except ImportError:
            _method = "pca"

    if _method in ("pca", "tsne"):
        _X = _build_features(designs_df)
        _coords, _label = _compute_embedding(
            _X, _method, perplexity=30.0, seed=42, n=len(designs_df)
        )

    embedding_df = pd.DataFrame(
        {
            "idx": np.arange(len(designs_df)),
            "x": _coords[:, 0],
            "y": _coords[:, 1],
            **{col: values_df[col].values for col in values_df.columns},
        }
    )
    return (embedding_df,)


@app.cell
def _(alt, color_by, embedding_df, mo, values_df):
    # Create scatter chart — displayed in the linked cell below, not here.
    scatter_chart = None

    mo.stop(
        embedding_df is None,
        mo.callout(
            mo.md("Load data and compute the embedding to see the plot."), kind="info"
        ),
    )

    _brush = alt.selection_interval(name="brush")
    _color_col = color_by.value if color_by.value in embedding_df.columns else None

    _color_enc = (
        alt.condition(
            _brush,
            alt.Color(
                f"{_color_col}:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=_color_col),
            ),
            alt.value("#cccccc"),
        )
        if _color_col
        else alt.condition(_brush, alt.value("steelblue"), alt.value("#cccccc"))
    )
    _obj_cols = list(values_df.columns) if values_df is not None else []
    _tooltip = ["idx:Q"] + [f"{c}:Q" for c in _obj_cols if c in embedding_df.columns]

    _spec = (
        alt.Chart(embedding_df)
        .mark_circle(size=40)
        .encode(
            x=alt.X("x:Q", title="Dim 1", axis=alt.Axis(grid=False)),
            y=alt.Y("y:Q", title="Dim 2", axis=alt.Axis(grid=False)),
            color=_color_enc,
            opacity=alt.condition(_brush, alt.value(0.9), alt.value(0.15)),
            tooltip=_tooltip,
        )
        .add_params(_brush)
        .properties(
            width=500, height=400, title="Pareto Front — Design Space Embedding"
        )
    )
    scatter_chart = mo.ui.altair_chart(_spec)
    return (scatter_chart,)


@app.cell
def _(ParallelCoordinates, embedding_df, mo, scatter_chart, values_df):
    # Read the current brush selection and filter the parallel coordinates to match.
    # When nothing is selected the full dataset is shown; when brushed only the
    # selected designs appear in the PC — keeping both views in sync.
    _selected = scatter_chart.value
    _n_total = len(embedding_df)
    _n_selected = len(_selected) if _selected is not None else 0
    _is_filtered = 0 < _n_selected < _n_total

    _val_cols = list(values_df.columns)
    _pc_data = _selected[_val_cols] if _is_filtered else values_df
    _pc_widget = mo.ui.anywidget(ParallelCoordinates(_pc_data, height=400))

    _label = (
        mo.md(
            f"*{_n_selected} of {_n_total} designs selected — parallel coordinates filtered to selection*"
        )
        if _is_filtered
        else mo.md(f"*{_n_total} designs — drag on scatter to filter*")
    )

    mo.vstack(
        [
            mo.hstack([scatter_chart, _pc_widget], gap="2rem"),
            _label,
        ]
    )


@app.cell
def _(
    designs_df,
    embedding_df,
    materials,
    mo,
    plot_coating_stack,
    plt,
    scatter_chart,
    values_df,
):
    if scatter_chart is None or designs_df is None or embedding_df is None:
        mo.stop()

    _selected = scatter_chart.value
    _n_total = len(embedding_df)
    _n_selected = len(_selected) if _selected is not None else 0
    _is_filtered = 0 < _n_selected < _n_total

    mo.stop(
        not _is_filtered,
        mo.callout(
            mo.md(
                "**Drag a selection box** on the scatter plot to view coating stacks."
            ),
            kind="info",
        ),
    )

    _sel_idxs = _selected["idx"].values.astype(int)
    _n_show = min(8, len(_sel_idxs))
    _obj_cols = list(values_df.columns)

    # Get thickness/material column lists from designs_df (exact matching)
    _thick_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("thickness_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )
    _mat_cols = sorted(
        [
            c
            for c in designs_df.columns
            if c.startswith("material_") and c.split("_")[1].isdigit()
        ],
        key=lambda c: int(c.split("_")[1]),
    )

    _fig, _axes = plt.subplots(1, _n_show, figsize=(_n_show * 2.5, 6), squeeze=False)

    for _ci, _ri in enumerate(_sel_idxs[:_n_show]):
        _ax = _axes[0, _ci]
        _row = designs_df.iloc[_ri]
        _thicknesses = _row[_thick_cols].values.astype(float)
        _mat_indices = _row[_mat_cols].values.astype(int)

        plot_coating_stack(
            thicknesses=_thicknesses,
            material_indices=_mat_indices,
            materials=materials,
            ax=_ax,
            convert_to_nm=True,
        )

        _parts = [f"{c}={values_df.iloc[_ri][c]:.3f}" for c in _obj_cols[:2]]
        _ax.set_title("\n".join(_parts), fontsize=7, pad=3)
        if _ci > 0:
            _ax.set_ylabel("")
            _ax.set_yticklabels([])

    _fig.suptitle(
        f"Coating stacks — {_n_show} of {_n_selected} selected",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
