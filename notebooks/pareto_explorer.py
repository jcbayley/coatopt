import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import altair as alt
    from pathlib import Path

    from coatopt.utils.utils import load_pareto_front, load_materials
    from coatopt.utils.plotting import plot_coating_stack

    return (
        Path,
        alt,
        json,
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
    **Step 2** — toggle between physical values and normalised rewards.
    **Step 3** — click *Compute embedding* to project the design space.
    **Step 4** — drag a selection in the **embedding** or **Pareto front** — both charts cross-filter each other, and the parallel coordinates reacts to both.
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
    rewards_toggle = mo.ui.switch(
        label="Show normalised rewards (instead of physical values)"
    )
    mo.vstack(
        [
            mo.hstack([file_browser, embedding_method], gap="2rem", align="start"),
            rewards_toggle,
        ]
    )
    return embedding_method, file_browser


@app.cell
def _(Path, file_browser, json, load_materials, load_pareto_front, mo):
    _val = file_browser.value
    _csv = Path(_val[0].path) if _val else None
    _run_dir = _csv.parent if _csv else None

    designs_df = None
    values_df = None
    rewards_df = None
    materials = {}
    run_metadata = None

    if _run_dir and (_run_dir / "pareto_front.csv").exists():
        try:
            _designs, _values, _rewards = load_pareto_front(_run_dir)
            _mask = ~_designs.duplicated()
            designs_df = _designs[_mask].reset_index(drop=True)
            values_df = _values[_mask].reset_index(drop=True)
            rewards_df = _rewards[_mask].reset_index(drop=True)
        except Exception:
            designs_df = None
            values_df = None
            rewards_df = None

        for _search_dir in [_run_dir] + list(_run_dir.parents)[:5]:
            _candidates = sorted(_search_dir.glob("materials*.json"))
            if _candidates:
                materials = load_materials(str(_candidates[0]))
                break

        _meta_path = _run_dir / "run_metadata.json"
        if _meta_path.exists():
            try:
                with open(_meta_path) as _f:
                    run_metadata = json.load(_f)
            except Exception("Warning: No metadata"):
                run_metadata = None

    if designs_df is not None:
        _n = len(designs_df)
        _objs = list(values_df.columns)
        _mat_str = (
            ", ".join(v["name"] for k, v in materials.items() if k > 0)
            if materials else "—"
        )
        _thick_cols = [
            c for c in designs_df.columns
            if c.startswith("thickness_") and c.split("_")[1].isdigit()
        ]
        _n_layers = len(_thick_cols)
        _items = [
            mo.callout(
                mo.md(
                    f"**{_n} designs** &nbsp;·&nbsp; {_n_layers} layers &nbsp;·&nbsp; "
                    f"objectives: `{'`, `'.join(_objs)}` &nbsp;·&nbsp; materials: {_mat_str}"
                ),
                kind="success",
            )
        ]
        if run_metadata:
            _algo = run_metadata.get("algorithm", "—")
            _dur_m = run_metadata.get("duration_minutes")
            _dur_h = run_metadata.get("duration_hours")
            _dur_str = (
                f"{_dur_h:.2f} h" if _dur_h and _dur_h >= 1
                else f"{_dur_m:.1f} min" if _dur_m else "—"
            )
            _gens = run_metadata.get("total_generations")
            _eps = run_metadata.get("total_episodes")
            _iters = (
                f"{_gens:,} generations" if _gens
                else f"{_eps:,} episodes" if _eps
                else "—"
            )
            _git = run_metadata.get("git_hash", "—")
            _pop = run_metadata.get("population_size")
            _pop_str = f"&nbsp;·&nbsp; population {_pop:,}" if _pop else ""
            _items.append(
                mo.callout(
                    mo.md(
                        f"**{_algo}** &nbsp;·&nbsp; {_iters}{_pop_str}"
                        f" &nbsp;·&nbsp; duration **{_dur_str}**"
                        f" &nbsp;·&nbsp; git `{_git}`"
                    ),
                    kind="info",
                )
            )
        mo.vstack(_items)
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
    return designs_df, materials, run_metadata


@app.cell
def _(run_metadata):
    run_metadata
    return


app._unparsable_cell(
    r"""
    |if rewards_toggle.value and rewards_df is not None:
        display_df = rewards_df
    elif values_df is not None:
        display_df = values_df
    else:
        display_df = None
    """,
    name="_"
)


@app.cell
def _(display_df, mo):
    _opts = list(display_df.columns) if display_df is not None else ["(none)"]
    color_by = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Color scatter by",
    )
    _pareto_opts = _opts if len(_opts) >= 2 else (_opts + _opts)[:2]
    pareto_x_col = mo.ui.dropdown(
        options=_pareto_opts,
        value=_pareto_opts[0],
        label="Pareto x-axis",
    )
    pareto_y_col = mo.ui.dropdown(
        options=_pareto_opts,
        value=_pareto_opts[1],
        label="Pareto y-axis",
    )
    mo.hstack([color_by, pareto_x_col, pareto_y_col], gap="1rem")
    return color_by, pareto_x_col, pareto_y_col


@app.cell
def _(mo):
    compute_btn = mo.ui.run_button(label="⚡  Compute embedding")
    compute_btn
    return (compute_btn,)


@app.cell
def _(compute_btn, designs_df, embedding_method, mo, np, pd):
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

    # Local import avoids marimo's _ prefix dependency-tracking issue
    from coatopt.utils.plot_design_diversity import (
        _build_features,
        _compute_embedding,
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
        except ImportError:
            _method = "pca"

    if _method in ("pca", "tsne"):
        _X = _build_features(designs_df)
        _coords, _ = _compute_embedding(
            _X, _method, perplexity=30.0, seed=42, n=len(designs_df)
        )

    embedding_df = pd.DataFrame(
        {
            "idx": np.arange(len(designs_df)),
            "emb_x": _coords[:, 0],
            "emb_y": _coords[:, 1],
        }
    )
    return (embedding_df,)


@app.cell
def _(
    alt,
    color_by,
    display_df,
    embedding_df,
    mo,
    np,
    pareto_x_col,
    pareto_y_col,
):
    combined_chart = None

    mo.stop(
        embedding_df is None,
        mo.callout(
            mo.md("Load data and compute the embedding to see the plots."), kind="info"
        ),
    )

    _pc_cols = list(display_df.columns) if display_df is not None else []
    _xcol = (
        pareto_x_col.value
        if pareto_x_col.value in _pc_cols
        else (_pc_cols[0] if _pc_cols else "")
    )
    _ycol = (
        pareto_y_col.value
        if pareto_y_col.value in _pc_cols
        else (_pc_cols[1] if len(_pc_cols) > 1 else _xcol)
    )

    # Single base dataframe shared by all three chart views.
    # Must contain embedding coords, pareto coords, and display columns
    # so that each cross-filter brush can operate on the correct axes.
    _base = embedding_df.copy()
    if display_df is not None:
        for _col in display_df.columns:
            _base[_col] = display_df[_col].values
    _base["par_x"] = _base[_xcol] if _xcol in _base.columns else 0.0
    _base["par_y"] = _base[_ycol] if _ycol in _base.columns else 0.0

    # Pre-normalise objective columns for the PC using global min/max
    # so lines don't jump when the selection changes.
    _n_rows = len(_base)
    for _col in _pc_cols:
        _vals = _base[_col].values.astype(float)
        _mn, _mx = _vals.min(), _vals.max()
        _base[f"__n_{_col}"] = (
            (_vals - _mn) / (_mx - _mn) if _mx > _mn else np.full(_n_rows, 0.5)
        )

    # Three independent selections — Vega-Lite 5 empty-selection semantics:
    # an empty selection evaluates to TRUE (matches all data), so the AND
    # combination below gives "all highlighted" when nothing is selected, and
    # narrows to the intersection as brushes become active.
    #
    # brush_emb  — interval drag on the embedding scatter
    # brush_par  — interval drag on the Pareto front
    # brush_pc   — click (toggle) individual lines in the parallel coordinates
    _brush_emb = alt.selection_interval(name="brush_emb")
    _brush_par = alt.selection_interval(name="brush_par")
    _brush_pc = alt.selection_point(fields=["idx"], name="brush_pc", toggle=True)

    # Unified condition shared by all three charts.
    # Works on any dataset that has emb_x/emb_y, par_x/par_y, and idx columns.
    _cond = _brush_emb & _brush_par & _brush_pc

    _cc = color_by.value if color_by.value in _base.columns else None
    _active_color = (
        alt.Color(
            f"{_cc}:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=alt.Legend(title=_cc),
        )
        if _cc
        else alt.value("steelblue")
    )

    _tip = ["idx:Q"] + [f"{c}:Q" for c in _pc_cols]

    # ── Embedding scatter ───────────────────────────────────────────────────
    _scatter = (
        alt.Chart(_base)
        .mark_circle(size=40)
        .encode(
            x=alt.X("emb_x:Q", title="Dim 1", axis=alt.Axis(grid=False)),
            y=alt.Y("emb_y:Q", title="Dim 2", axis=alt.Axis(grid=False)),
            color=alt.condition(_cond, _active_color, alt.value("#cccccc")),
            opacity=alt.condition(_cond, alt.value(0.9), alt.value(0.15)),
            tooltip=_tip,
        )
        .add_params(_brush_emb)
        .properties(title="Design Space Embedding", width=380, height=330)
    )

    # ── Pareto front ────────────────────────────────────────────────────────
    _pareto = (
        alt.Chart(_base)
        .mark_circle(size=50)
        .encode(
            x=alt.X("par_x:Q", title=_xcol),
            y=alt.Y("par_y:Q", title=_ycol),
            color=alt.condition(_cond, alt.value("steelblue"), alt.value("#cccccc")),
            opacity=alt.condition(_cond, alt.value(0.9), alt.value(0.15)),
            tooltip=_tip,
        )
        .add_params(_brush_par)
        .properties(title=f"Pareto Front  ({_xcol} vs {_ycol})", width=380, height=330)
    )

    # ── Parallel coordinates ────────────────────────────────────────────────
    # transform_fold runs AFTER the data is loaded; the base df already carries
    # emb_x/emb_y/par_x/par_y/idx so brush_emb, brush_par, brush_pc all
    # evaluate correctly on the folded rows.
    _norm_cols = [f"__n_{c}" for c in _pc_cols]
    _axis_lookup = "{" + ", ".join(f'"__n_{c}": "{c}"' for c in _pc_cols) + "}"

    # Colour active lines by the same objective as the scatter; grey out the rest.
    _pc_color = alt.condition(
        _cond,
        alt.Color(f"{_cc}:Q", scale=alt.Scale(scheme="viridis"), legend=None)
        if _cc
        else alt.value("steelblue"),
        alt.value("#dddddd"),
    )

    _pc = (
        alt.Chart(_base)
        .transform_fold(_norm_cols, as_=["__axis_key", "norm_value"])
        .transform_calculate(axis=f"{_axis_lookup}[datum.__axis_key]")
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("axis:N", title=None, sort=_pc_cols),
            y=alt.Y(
                "norm_value:Q",
                title="Normalised value",
                scale=alt.Scale(domain=[0, 1]),
            ),
            detail="idx:N",
            color=_pc_color,
            opacity=alt.condition(_cond, alt.value(0.7), alt.value(0.08)),
            tooltip=_tip,
        )
        .add_params(_brush_pc)
        .properties(
            title="Parallel Coordinates  (click lines to select; drag boxes above to filter)",
            width=800,
            height=260,
        )
    )

    _spec = alt.vconcat(
        alt.hconcat(_scatter, _pareto),
        _pc,
    )
    combined_chart = mo.ui.altair_chart(_spec)
    return (combined_chart,)


@app.cell
def _(combined_chart, embedding_df, mo):
    _n_total = len(embedding_df) if embedding_df is not None else 0
    _sel = combined_chart.value if combined_chart is not None else None
    _n_sel = len(_sel) if _sel is not None else 0
    _label = (
        mo.md(f"*{_n_sel} of {_n_total} designs selected*")
        if 0 < _n_sel < _n_total
        else mo.md(
            f"*{_n_total} designs — drag a region in the embedding or Pareto front, or click lines in the parallel coordinates*"
        )
    )
    mo.vstack([combined_chart, _label])
    return


@app.cell
def _(
    combined_chart,
    designs_df,
    display_df,
    embedding_df,
    materials,
    mo,
    plot_coating_stack,
    plt,
):
    if combined_chart is None or designs_df is None or embedding_df is None:
        mo.stop()

    _selected = combined_chart.value
    _n_total = len(embedding_df)
    _n_selected = len(_selected) if _selected is not None else 0
    _is_filtered = 0 < _n_selected < _n_total

    mo.stop(
        not _is_filtered,
        mo.callout(
            mo.md(
                "**Drag a selection box** on the embedding or Pareto chart to view coating stacks."
            ),
            kind="info",
        ),
    )

    _sel_idxs = _selected["idx"].values.astype(int)
    _n_show = min(8, len(_sel_idxs))
    _obj_cols = list(display_df.columns) if display_df is not None else []

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

        _parts = (
            [f"{c}={display_df.iloc[_ri][c]:.3f}" for c in _obj_cols[:2]]
            if display_df is not None
            else []
        )
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
