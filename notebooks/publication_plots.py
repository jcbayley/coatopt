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
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    from coatopt.utils.utils import load_pareto_front, load_materials
    from coatopt.utils.plotting import plot_coating_stack

    return (
        Path,
        gridspec,
        json,
        load_materials,
        load_pareto_front,
        mo,
        np,
        plot_coating_stack,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Publication Plot Generator

    Load a Pareto front run directory, pick a plot type and configure it, then save.

    **Available plot types:**
    - **Scatter matrix** — lower-left pairwise trade-offs with Pareto boundary; standard for multi-objective papers
    - **Parallel coordinates** — each design as a polyline; shows clustering and compromise regions
    - **Radar / spider chart** — normalised objective profiles; good for comparing solution diversity
    - **Objective distributions** — violin + jitter; quantifies the achievable range of each objective
    - **Featured 2D scatter** — clean single panel with colour and optional size encoding
    - **Design showcase** — coating stack plots for the most extreme design in each objective
    - **Design Inspector** — scatter matrix with highlighted designs + coating stacks below; enter indices in the Inspector field
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
    file_browser
    return (file_browser,)


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
            except Exception:
                run_metadata = None

    if designs_df is not None:
        _n = len(designs_df)
        _objs = list(values_df.columns)
        _items = [
            mo.callout(
                mo.md(
                    f"**{_n} Pareto designs** &nbsp;·&nbsp; "
                    f"objectives: `{'`, `'.join(_objs)}`"
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
    elif _run_dir:
        mo.callout(
            mo.md(f"No `pareto_front.csv` found in `{_run_dir.name}`."),
            kind="warn",
        )
    else:
        mo.callout(mo.md("Select a `pareto_front.csv` above."), kind="warn")
    return designs_df, materials, rewards_df, run_metadata, values_df


@app.cell
def _(mo):
    rewards_toggle = mo.ui.switch(
        label="Use normalised rewards (instead of physical values)"
    )
    return (rewards_toggle,)


@app.cell
def _(rewards_df, rewards_toggle, values_df):

    display_df = (
        rewards_df
        if (rewards_toggle.value and rewards_df is not None)
        else values_df
    )
    rewards_toggle
    return (display_df,)


@app.cell
def _(display_df, mo):
    _opts = list(display_df.columns) if display_df is not None else ["(none)"]
    _pareto_opts = (_opts * 2)[:max(len(_opts), 2)]

    plot_type = mo.ui.dropdown(
        options={
            "Scatter matrix": "scatter_matrix",
            "Parallel coordinates": "parallel_coords",
            "Radar / spider chart": "radar",
            "Objective distributions": "distributions",
            "Featured 2D scatter": "scatter_2d",
            "Design showcase": "showcase",
            "Design Inspector": "inspector",
        },
        value="Scatter matrix",
        label="Plot type",
    )
    color_obj = mo.ui.dropdown(options=_opts, value=_opts[0], label="Colour by")

    scatter_x = mo.ui.dropdown(options=_pareto_opts, value=_pareto_opts[0], label="2D x-axis")
    scatter_y = mo.ui.dropdown(options=_pareto_opts, value=_pareto_opts[1], label="2D y-axis")
    scatter_size = mo.ui.dropdown(
        options=["(none)"] + _opts, value="(none)", label="2D size by"
    )

    mo.vstack([
        mo.hstack([plot_type, color_obj], gap="2rem", align="start"),
        mo.hstack([scatter_x, scatter_y, scatter_size], gap="1rem"),
    ])
    return color_obj, plot_type, scatter_size, scatter_x, scatter_y


@app.cell
def _(mo):
    dpi = mo.ui.slider(start=72, stop=300, step=6, value=150, label="DPI")
    fig_width = mo.ui.slider(
        start=3.5, stop=14.0, step=0.5, value=8.0, label="Width (in)"
    )
    fig_format = mo.ui.dropdown(
        options=["png", "pdf", "svg"], value="png", label="Format"
    )
    cmap_choice = mo.ui.dropdown(
        options={
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Cividis (CB-safe)": "cividis",
            "Coolwarm": "coolwarm",
        },
        value="Viridis",
        label="Colourmap",
    )
    design_indices_text = mo.ui.text(
        value="0, 1, 2",
        label="Inspector indices (comma-separated)",
    )
    inspector_layout = mo.ui.dropdown(
        options={"Stacks below": "below", "Stacks right": "right"},
        value="Stacks below",
        label="Inspector layout",
    )
    save_dir_text = mo.ui.text(value="./figures", label="Save directory")
    save_btn = mo.ui.run_button(label="💾  Save figure")

    mo.vstack([
        mo.hstack([dpi, fig_width, cmap_choice], gap="1rem"),
        mo.hstack([design_indices_text, inspector_layout, fig_format, save_dir_text, save_btn], gap="1rem", align="end"),
    ])
    return (
        cmap_choice,
        design_indices_text,
        dpi,
        fig_format,
        fig_width,
        inspector_layout,
        save_btn,
        save_dir_text,
    )


@app.cell
def _(
    Path,
    cmap_choice,
    color_obj,
    design_indices_text,
    designs_df,
    display_df,
    dpi,
    fig_format,
    fig_width,
    gridspec,
    inspector_layout,
    materials,
    mo,
    np,
    plot_coating_stack,
    plot_type,
    plt,
    rewards_toggle,
    save_btn,
    save_dir_text,
    scatter_size,
    scatter_x,
    scatter_y,
):
    mo.stop(
        display_df is None,
        mo.callout(mo.md("Load a run directory first."), kind="warn"),
    )

    _cols = list(display_df.columns)
    _n_obj = len(_cols)
    _fw = fig_width.value
    _dpi = dpi.value
    _cmap = plt.get_cmap(cmap_choice.value)
    _cc = color_obj.value if color_obj.value in _cols else _cols[0]
    _cvals = display_df[_cc].values.astype(float)
    _cnorm = plt.Normalize(_cvals.min(), _cvals.max())
    _colors = _cmap(_cnorm(_cvals))
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_cnorm)
    _sm.set_array([])

    def _pareto_2d(xv, yv, minimize=True):
        """Return (px, py) of 2D Pareto-front points sorted by x."""
        pts = np.column_stack([xv.astype(float), yv.astype(float)])
        if not minimize:
            pts = -pts
        mask = np.ones(len(pts), dtype=bool)
        for _i in range(len(pts)):
            if mask[_i]:
                dom = np.all(pts <= pts[_i], axis=1) & np.any(pts < pts[_i], axis=1)
                dom[_i] = False
                if dom.any():
                    mask[_i] = False
        front = pts[mask]
        sx = np.argsort(front[:, 0])
        px, py = front[sx, 0], front[sx, 1]
        if not minimize:
            px, py = -px, -py
            sx2 = np.argsort(px)
            px, py = px[sx2], py[sx2]
        return px, py

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": _dpi,
    })

    _pt = plot_type.value
    fig = None

    # ── Scatter matrix (lower-left triangle only) ────────────────────────────
    if _pt == "scatter_matrix":
        _minimize = not rewards_toggle.value
        fig, _axes = plt.subplots(
            _n_obj, _n_obj,
            figsize=(_fw, _fw * 0.92),
            squeeze=False,
        )
        _visible_axes = []
        for _r in range(_n_obj):
            for _c in range(_n_obj):
                _ax = _axes[_r][_c]
                if _c > _r:  # upper-right triangle — hide
                    _ax.set_visible(False)
                    continue
                _visible_axes.append(_ax)
                if _r == _c:
                    _ax.hist(
                        display_df[_cols[_r]], bins=22,
                        color="steelblue", alpha=0.72,
                        edgecolor="white", linewidth=0.3,
                    )
                else:
                    _ax.scatter(
                        display_df[_cols[_c]], display_df[_cols[_r]],
                        c=_colors, s=12, alpha=0.65, linewidths=0,
                    )
                    try:
                        _px, _py = _pareto_2d(
                            display_df[_cols[_c]].values,
                            display_df[_cols[_r]].values,
                            minimize=_minimize,
                        )
                        _ax.step(
                            _px, _py, where="post",
                            color="crimson", linewidth=1.0,
                            linestyle="--", alpha=0.75, zorder=5,
                        )
                    except Exception:
                        pass
                if _r == _n_obj - 1:
                    _ax.set_xlabel(_cols[_c], fontsize=9)
                else:
                    _ax.set_xticklabels([])
                if _c == 0:
                    _ax.set_ylabel("Count" if _r == _c else _cols[_r], fontsize=9)
                else:
                    _ax.set_yticklabels([])
        _cbar = fig.colorbar(_sm, ax=_visible_axes, shrink=0.55, pad=0.02)
        _cbar.set_label(_cc, fontsize=9)
        fig.suptitle("Pareto Front — Pairwise Scatter Matrix", fontsize=12, y=1.01)
        plt.tight_layout()

    # ── Parallel coordinates ─────────────────────────────────────────────────
    elif _pt == "parallel_coords":
        fig, _ax = plt.subplots(figsize=(_fw, _fw * 0.52))

        _ndf = {}
        _axis_range = {}
        for _col in _cols:
            _mn, _mx = display_df[_col].min(), display_df[_col].max()
            _rng = _mx - _mn if _mx > _mn else 1.0
            _ndf[_col] = (display_df[_col].values - _mn) / _rng
            _axis_range[_col] = (_mn, _mx)

        _x = np.arange(_n_obj, dtype=float)
        _sort_idx = np.argsort(_cvals)
        for _i in _sort_idx:
            _ax.plot(
                _x, [_ndf[c][_i] for c in _cols],
                color=_colors[_i], alpha=0.35, linewidth=0.85,
                solid_capstyle="round",
            )

        for _xi, _col in enumerate(_cols):
            _ax.axvline(_xi, color="#555", linewidth=0.9, alpha=0.55)
            _mn, _mx = _axis_range[_col]
            _ax.text(_xi, -0.10, f"{_mn:.3g}", ha="center", va="top",
                     fontsize=8, color="#555")
            _ax.text(_xi, 1.07, f"{_mx:.3g}", ha="center", va="bottom",
                     fontsize=8, color="#555")

        _ax.set_xticks(_x)
        _ax.set_xticklabels(_cols, fontsize=10)
        _ax.set_ylabel("Normalised value", fontsize=11)
        _ax.set_ylim(-0.06, 1.06)
        _ax.set_xlim(-0.2, _n_obj - 0.8)
        _ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        _ax.spines["bottom"].set_visible(False)
        _ax.spines["left"].set_visible(False)
        fig.colorbar(_sm, ax=_ax, pad=0.02).set_label(_cc, fontsize=10)
        _ax.set_title("Pareto Front — Parallel Coordinates", fontsize=12)
        fig.tight_layout()

    # ── Radar / spider chart ─────────────────────────────────────────────────
    elif _pt == "radar":
        if _n_obj < 3:
            fig, _ax = plt.subplots(figsize=(_fw * 0.7, _fw * 0.7))
            _ax.text(0.5, 0.5, "Radar chart needs ≥ 3 objectives",
                     ha="center", va="center", transform=_ax.transAxes, fontsize=12)
            _ax.axis("off")
        else:
            _ndf = {}
            for _col in _cols:
                _mn, _mx = display_df[_col].min(), display_df[_col].max()
                _rng = _mx - _mn if _mx > _mn else 1.0
                _ndf[_col] = (display_df[_col].values - _mn) / _rng

            _angles = np.linspace(0, 2 * np.pi, _n_obj, endpoint=False)
            _angles_c = np.append(_angles, _angles[0])

            fig, _ax = plt.subplots(
                figsize=(_fw * 0.72, _fw * 0.72),
                subplot_kw={"polar": True},
            )
            _n_draw = min(len(display_df), 80)
            _step = max(1, len(display_df) // _n_draw)
            _sort_idx = np.argsort(_cvals)
            for _i in _sort_idx[::_step]:
                _v = np.append([_ndf[c][_i] for c in _cols], _ndf[_cols[0]][_i])
                _ax.plot(_angles_c, _v, color=_colors[_i], alpha=0.28, linewidth=0.9)
                _ax.fill(_angles_c, _v, color=_colors[_i], alpha=0.04)

            _ax.set_thetagrids(np.degrees(_angles), _cols, fontsize=10)
            _ax.set_rlim(0, 1)
            _ax.set_rticks([0.25, 0.5, 0.75, 1.0])
            _ax.set_rlabel_position(10)
            _ax.set_title(
                "Pareto Front — Radar Chart\n(normalised objectives)",
                fontsize=12, pad=18,
            )
            _cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            fig.colorbar(_sm, cax=_cbar_ax).set_ylabel(_cc, fontsize=10)
        fig.tight_layout()

    # ── Objective distributions ───────────────────────────────────────────────
    elif _pt == "distributions":
        fig, _axes = plt.subplots(
            1, _n_obj, figsize=(_fw, _fw * 0.48), sharey=False
        )
        if _n_obj == 1:
            _axes = [_axes]
        for _i, (_col, _ax) in enumerate(zip(_cols, _axes)):
            _data = display_df[_col].values.astype(float)
            _vp = _ax.violinplot([_data], positions=[0], showmedians=True, showextrema=True)
            for _pc in _vp["bodies"]:
                _pc.set_facecolor("steelblue")
                _pc.set_alpha(0.35)
            _vp["cmedians"].set_color("white")
            _vp["cmedians"].set_linewidth(2.0)
            _vp["cmins"].set_linewidth(0.8)
            _vp["cmaxes"].set_linewidth(0.8)
            _vp["cbars"].set_linewidth(0.8)
            _ax.scatter(
                np.random.default_rng(42).uniform(-0.08, 0.08, len(_data)),
                _data,
                c=_colors,
                s=9, alpha=0.6, zorder=3, linewidths=0,
            )
            _ax.set_xticks([])
            _ax.set_xlabel(_col, fontsize=10, labelpad=6)
            if _i == 0:
                _ax.set_ylabel("Value", fontsize=11)
            else:
                _ax.set_yticklabels([])
            _med = np.median(_data)
            _ax.axhline(_med, color="crimson", linewidth=1.0, linestyle="--", alpha=0.75)
            _ax.text(
                0.5, 0.98,
                f"med {_med:.3g}\n[{_data.min():.3g}, {_data.max():.3g}]",
                ha="center", va="top", transform=_ax.transAxes,
                fontsize=7.5, color="#555",
            )
        fig.colorbar(
            _sm,
            ax=_axes if _n_obj > 1 else _axes[0],
            shrink=0.85, pad=0.01,
        ).set_label(_cc, fontsize=10)
        fig.suptitle("Pareto Front — Objective Distributions", fontsize=12)
        fig.tight_layout()

    # ── Featured 2D scatter ───────────────────────────────────────────────────
    elif _pt == "scatter_2d":
        _xcol = scatter_x.value if scatter_x.value in _cols else _cols[0]
        _ycol = scatter_y.value if scatter_y.value in _cols else _cols[min(1, _n_obj - 1)]
        _szc = scatter_size.value if scatter_size.value in _cols else None

        _sizes = None
        if _szc:
            _sv = display_df[_szc].values.astype(float)
            _rng = _sv.max() - _sv.min()
            _sizes = 20 + 180 * (_sv - _sv.min()) / (_rng if _rng > 0 else 1.0)

        fig, _ax = plt.subplots(figsize=(_fw * 0.78, _fw * 0.65))
        _ax.scatter(
            display_df[_xcol], display_df[_ycol],
            c=_colors,
            s=_sizes if _sizes is not None else 30,
            alpha=0.75,
            linewidths=0.3,
            edgecolors="white",
        )
        _ax.set_xlabel(_xcol, fontsize=12)
        _ax.set_ylabel(_ycol, fontsize=12)
        _title = f"Pareto Front  —  {_xcol} vs {_ycol}"
        if _szc:
            _title += f"   (size ∝ {_szc})"
        _ax.set_title(_title, fontsize=12)
        fig.colorbar(_sm, ax=_ax, pad=0.02).set_label(_cc, fontsize=10)
        fig.tight_layout()

    # ── Design showcase ───────────────────────────────────────────────────────
    elif _pt == "showcase":
        # Show the design that is most extreme in EACH objective (best/worst).
        # For each objective, pick the design at the 5th-percentile value
        # (assumed to be optimized, i.e. low is good — flip for user).
        if designs_df is None:
            fig, _ax = plt.subplots()
            _ax.text(0.5, 0.5, "Design showcase needs designs to be loaded.",
                     ha="center", va="center", transform=_ax.transAxes)
            _ax.axis("off")
        else:
            _thick_cols = sorted(
                [c for c in designs_df.columns
                 if c.startswith("thickness_") and c.split("_")[1].isdigit()],
                key=lambda c: int(c.split("_")[1]),
            )
            _mat_cols = sorted(
                [c for c in designs_df.columns
                 if c.startswith("material_") and c.split("_")[1].isdigit()],
                key=lambda c: int(c.split("_")[1]),
            )
            # Pick the index that minimises each objective
            _showcase_idx = [int(display_df[c].idxmin()) for c in _cols]
            # Also add the index that is the Pareto "knee" (closest to ideal)
            _ndf2 = (display_df - display_df.min()) / (display_df.max() - display_df.min() + 1e-12)
            _knee_idx = int(np.linalg.norm(_ndf2.values, axis=1).argmin())
            _all_idx = list(dict.fromkeys(_showcase_idx + [_knee_idx]))  # unique, order-preserving

            _n_show = len(_all_idx)
            _labels = [f"Best {_cols[i]}" for i in range(_n_obj)] + ["Knee"]
            _labels = _labels[:_n_show]

            fig, _axes = plt.subplots(
                1, _n_show,
                figsize=(_n_show * 2.4, 6),
                squeeze=False,
            )
            for _ci, (_ri, _lbl) in enumerate(zip(_all_idx, _labels)):
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
                _obj_strs = [f"{c}={display_df.iloc[_ri][c]:.3g}" for c in _cols]
                _ax.set_title(
                    _lbl + "\n" + "\n".join(_obj_strs),
                    fontsize=7, pad=3,
                )
                if _ci > 0:
                    _ax.set_ylabel("")
                    _ax.set_yticklabels([])
            fig.suptitle(
                "Showcase — extreme and knee designs",
                fontsize=12, y=1.02,
            )
            plt.tight_layout()

    # ── Design Inspector ─────────────────────────────────────────────────────
    elif _pt == "inspector":
        _minimize = not rewards_toggle.value
        _layout = inspector_layout.value  # "below" or "right"
        try:
            _sel_idx = [
                int(x.strip())
                for x in design_indices_text.value.split(",")
                if x.strip().lstrip("-").isdigit()
            ]
            _sel_idx = [i for i in _sel_idx if 0 <= i < len(display_df)]
        except Exception:
            _sel_idx = []

        if designs_df is None:
            fig, _ax = plt.subplots(figsize=(_fw, 3))
            _ax.text(0.5, 0.5, "Inspector needs designs to be loaded.",
                     ha="center", va="center", transform=_ax.transAxes, fontsize=11)
            _ax.axis("off")
        else:
            _n_show = max(1, len(_sel_idx))
            _thick_cols = sorted(
                [c for c in designs_df.columns
                 if c.startswith("thickness_") and c.split("_")[1].isdigit()],
                key=lambda c: int(c.split("_")[1]),
            )
            _mat_cols_d = sorted(
                [c for c in designs_df.columns
                 if c.startswith("material_") and c.split("_")[1].isdigit()],
                key=lambda c: int(c.split("_")[1]),
            )

            # ── Build figure + gridspec for the chosen layout ─────────────────
            if _layout == "below":
                _mat_h = max(4.0, _n_obj * 1.8)
                _stk_h = max(3.5, _n_show * 0.6)
                fig = plt.figure(figsize=(_fw, _mat_h + _stk_h))
                _gs = gridspec.GridSpec(
                    2, 1, height_ratios=[_mat_h, _stk_h],
                    hspace=0.4, figure=fig,
                )
                _gs_mat = gridspec.GridSpecFromSubplotSpec(
                    _n_obj, _n_obj, subplot_spec=_gs[0], hspace=0.08, wspace=0.08,
                )
                _gs_stk = gridspec.GridSpecFromSubplotSpec(
                    1, _n_show, subplot_spec=_gs[1], wspace=0.15,
                )
                def _stack_ax(ci):
                    return fig.add_subplot(_gs_stk[0, ci])
                def _empty_stk_ax():
                    return fig.add_subplot(_gs_stk[0, :])
            else:  # "right"
                _mat_w = max(3.0, _n_obj * 1.5)
                _stk_w = max(2.0, _n_show * 1.8)
                _fig_h = max(_fw * 0.85, _n_obj * 1.5)
                fig = plt.figure(figsize=(_fw, _fig_h))
                _gs = gridspec.GridSpec(
                    1, 2, width_ratios=[_mat_w, _stk_w],
                    wspace=0.35, figure=fig,
                )
                _gs_mat = gridspec.GridSpecFromSubplotSpec(
                    _n_obj, _n_obj, subplot_spec=_gs[0], hspace=0.08, wspace=0.08,
                )
                _gs_stk = gridspec.GridSpecFromSubplotSpec(
                    _n_show, 1, subplot_spec=_gs[1], hspace=0.35,
                )
                def _stack_ax(ci):
                    return fig.add_subplot(_gs_stk[ci, 0])
                def _empty_stk_ax():
                    return fig.add_subplot(_gs_stk[:, 0])

            # ── Scatter matrix — lower-left, colored + highlighted ────────────
            for _r in range(_n_obj):
                for _c in range(_n_obj):
                    if _c > _r:
                        continue
                    _ax = fig.add_subplot(_gs_mat[_r, _c])
                    if _r == _c:
                        _ax.hist(
                            display_df[_cols[_r]], bins=22,
                            color="steelblue", alpha=0.55,
                            edgecolor="white", linewidth=0.3,
                        )
                        for _si in _sel_idx:
                            _ax.axvline(
                                display_df[_cols[_r]].iloc[_si],
                                color="crimson", linewidth=1.5, alpha=0.9,
                            )
                    else:
                        _ax.scatter(
                            display_df[_cols[_c]], display_df[_cols[_r]],
                            c=_colors, s=8, alpha=0.55, linewidths=0, zorder=1,
                        )
                        try:
                            _px, _py = _pareto_2d(
                                display_df[_cols[_c]].values,
                                display_df[_cols[_r]].values,
                                minimize=_minimize,
                            )
                            _ax.step(
                                _px, _py, where="post",
                                color="k", linewidth=0.9,
                                linestyle="--", alpha=0.6, zorder=4,
                            )
                        except Exception:
                            pass
                        for _si in _sel_idx:
                            _ax.scatter(
                                [display_df[_cols[_c]].iloc[_si]],
                                [display_df[_cols[_r]].iloc[_si]],
                                c="crimson", s=80, alpha=1.0, zorder=5,
                                linewidths=1.2, edgecolors="white",
                            )
                    if _r == _n_obj - 1:
                        _ax.set_xlabel(_cols[_c], fontsize=8)
                    else:
                        _ax.set_xticklabels([])
                    if _c == 0:
                        _ax.set_ylabel("Count" if _r == _c else _cols[_r], fontsize=8)
                    else:
                        _ax.set_yticklabels([])
                    _ax.tick_params(labelsize=7)

            # ── Coating stacks ────────────────────────────────────────────────
            if _sel_idx:
                for _ci, _si in enumerate(_sel_idx):
                    _ax = _stack_ax(_ci)
                    _row = designs_df.iloc[_si]
                    plot_coating_stack(
                        thicknesses=_row[_thick_cols].values.astype(float),
                        material_indices=_row[_mat_cols_d].values.astype(int),
                        materials=materials,
                        ax=_ax,
                        convert_to_nm=True,
                    )
                    _obj_strs = [f"{c}={display_df.iloc[_si][c]:.3g}" for c in _cols]
                    _ax.set_title(
                        f"Design #{_si}\n" + "\n".join(_obj_strs),
                        fontsize=7, pad=2,
                    )
                    if _layout == "below" and _ci > 0:
                        _ax.set_ylabel("")
                        _ax.set_yticklabels([])
            else:
                _ax = _empty_stk_ax()
                _ax.text(0.5, 0.5,
                         "Enter comma-separated design indices in the field above.",
                         ha="center", va="center",
                         transform=_ax.transAxes, fontsize=10)
                _ax.axis("off")

            fig.suptitle("Design Inspector", fontsize=12, y=1.01)

    # ── Save ─────────────────────────────────────────────────────────────────
    _save_msg = None
    if save_btn.value and fig is not None:
        _sd = Path(save_dir_text.value)
        _sd.mkdir(parents=True, exist_ok=True)
        _fname = f"pareto_{_pt}.{fig_format.value}"
        fig.savefig(_sd / _fname, dpi=_dpi, bbox_inches="tight")
        _save_msg = mo.callout(
            mo.md(f"Saved → `{_sd / _fname}`"), kind="success"
        )

    mo.vstack([x for x in [_save_msg, fig] if x is not None])
    return


if __name__ == "__main__":
    app.run()
