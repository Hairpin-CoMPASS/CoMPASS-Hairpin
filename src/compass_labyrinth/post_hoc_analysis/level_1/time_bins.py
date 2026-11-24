"""
TIME-BINNED MAZE HEATMAPS
Goal:
    ├── Label every analyzed frame with its time bin within its session:
    │     pct_bin   -- 5 equal parts of the session's own frames (0-20%, ..., 80-100%)
    │     frame_bin -- 0-10k, 10k-20k, ... 40k-50k frames (later frames: no bin)
    └── Draw any per-cell heatmap measure as one row per group x one column per bin.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compass_labyrinth.source_data as source_data

# report selector key -> (frame column, column titles)
BINNINGS = {
    "pct_bins": ("pct_bin", ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
    "frame_bins": ("frame_bin", ["0-10k frames", "10k-20k", "20k-30k", "30k-40k", "40k-50k"]),
}
BINNING_TITLES = {"pct_bins": "% of session", "frame_bins": "frames"}


def add_time_bins(df: pd.DataFrame, n_pct_bins: int = 5, frame_bin_size: int = 10000,
                  n_frame_bins: int = 5) -> pd.DataFrame:
    """`df` (analyzed frames with 'session' and 's_no') plus 'pct_bin' (1..n:
    which n-th of that session's own frames, in frame order) and 'frame_bin'
    (1..n: which frame_bin_size block of the session's frames; NaN past the
    last bin). Positions count analyzed frames (rows), like Plot 3's time bins."""
    out = df.copy()
    pos = out.groupby("session")["s_no"].rank(method="first") - 1
    n = out.groupby("session")["s_no"].transform("size")
    out["pct_bin"] = np.minimum((pos / n * n_pct_bins).astype(int) + 1, n_pct_bins)
    fb = (pos // frame_bin_size).astype(int) + 1
    out["frame_bin"] = fb.where(fb <= n_frame_bins)
    return out


def binned_cell_values(df: pd.DataFrame, bin_col: str, groups: list, values_fn) -> dict:
    """{(group, bin): Series(grid cell -> value)} -- `values_fn(frames)` turns
    one bin's frames (all groups) into a DataFrame with 'genotype',
    'grid_number' and 'value'."""
    out = {}
    for b in sorted(df[bin_col].dropna().unique()):
        vals = values_fn(df[df[bin_col] == b])
        for g in groups:
            sub = vals[vals["genotype"] == g]
            out[(g, int(b))] = sub.set_index("grid_number")["value"] if len(sub) else pd.Series(dtype=float)
    return out


def plot_binned_heatmaps(
    config: dict,
    values: dict,
    rows: list,
    binning: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
    title: str,
    cbar_label: str,
    grid_filename: str = "Session0001_grid.shp",
    row_n: dict | None = None,
    bin_n: dict | None = None,
    arrows_fn=None,
    diff_rows: list | None = None,
    record: pd.DataFrame | None = None,
    filename: str = "binned_heatmap",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """One row per entry of `rows` ((row key, row label)), one column per time
    bin of `binning` ("pct_bins" / "frame_bins"); each panel the maze colored
    by values[(row key, bin)] (a grid cell -> value Series), labeled with the
    values, grey where missing, reward outlined in gold; one shared colorbar.
    arrows_fn(ax, row_key, bin), if given, draws on each panel (Plot 7).
    bin_n {(row key, bin): mice with frames in that bin}: printed under a
    panel when fewer than the row's n (e.g. sessions shorter than 50k frames).
    diff_rows [(row key, row label, key B, key A)]: rows of B - A per bin,
    cell by cell, added below `rows` on a blue-white-red scale centered at 0
    (one shared limit: 95th percentile of all the figure's differences)."""
    import geopandas as gpd
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import (
        create_grid_geodata, hold_draws, release_draws, robust_diff_lim, DIFF_CMAP)

    _, bin_titles = BINNINGS[binning]
    grid = create_grid_geodata(config, grid_filename)
    reward = gpd.GeoDataFrame(geometry=[grid[grid["FID"].isin([70, 71, 80, 81])].union_all()], crs=grid.crs)
    # rows: (key, label) with the figure's colors, or (key, label, cmap, vmin, vmax) per row
    specs = [(r[0], r[1], *(r[2:] if len(r) == 5 else (cmap, vmin, vmax))) for r in rows]
    diff_keys = {d[0] for d in diff_rows or []}
    if diff_rows:
        diffs = {}
        for key, _, kb, ka in diff_rows:
            for b in range(1, len(bin_titles) + 1):
                if (kb, b) in values and (ka, b) in values:
                    diffs[(key, b)] = (values[(kb, b)] - values[(ka, b)]).dropna()
        lim = robust_diff_lim([s.to_numpy() for s in diffs.values()])
        values = {**values, **diffs}
        specs += [(key, label, DIFF_CMAP, -lim, lim) for key, label, *_ in diff_rows]
        if record is not None:
            record = pd.concat([record, values_record(diffs, binning)], ignore_index=True)
    fig, axes = plt.subplots(len(specs), len(bin_titles), figsize=(3.4 * len(bin_titles) + 1.4, 3.3 * len(specs) + 0.6),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    hold_draws(fig)
    for r, (key, label, r_cmap, r_lo, r_hi) in enumerate(specs):
        cm, norm = plt.get_cmap(r_cmap), plt.Normalize(r_lo, r_hi)
        for c in range(len(bin_titles)):
            ax = axes[r, c]
            v = values.get((key, c + 1), pd.Series(dtype=float))
            mapped = grid.assign(value=grid["FID"].map(v))
            mapped.plot(column="value", cmap=r_cmap, vmin=r_lo, vmax=r_hi, ax=ax, edgecolor="0.8", linewidth=0.5,
                        missing_kwds={"color": "0.85", "edgecolor": "0.8"})
            reward.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.4)
            if arrows_fn is None or key in diff_keys:
                for _, cell in mapped.iterrows():
                    val = cell["value"]
                    if pd.isna(val):
                        continue
                    rgba = cm(norm(val))
                    dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                    pt = cell.geometry.centroid
                    ax.annotate(fmt.format(val), (pt.x, pt.y), ha="center", va="center", fontsize=3.6,
                                color="white" if dark else "black")
            else:
                arrows_fn(ax, key, c + 1)
            ax.invert_yaxis()
            ax.axis("off")
            if bin_n and row_n and bin_n.get((key, c + 1), 0) < row_n.get(key, 0):
                ax.text(0.5, -0.01, f"n={bin_n.get((key, c + 1), 0)} of {row_n[key]}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8, color="0.3")
            if r == 0:
                ax.set_title(bin_titles[c], fontsize=11, weight="bold")
            if c == 0:
                txt = label + (f"\n(n={row_n[key]})" if row_n and key in row_n else "")
                ax.text(-0.05, 0.5, txt, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=10, weight="bold")
    # one colorbar per distinct color scale, beside the rows that use it (difference rows: their own)
    labels = cbar_label if isinstance(cbar_label, dict) else {}
    for scale in dict.fromkeys((sp[2], sp[3], sp[4], sp[0] in diff_keys) for sp in specs):
        rows_using = [i for i, sp in enumerate(specs) if (sp[2], sp[3], sp[4], sp[0] in diff_keys) == scale]
        text = "Difference (B − A)" if scale[3] else labels.get(
            scale[0], cbar_label if isinstance(cbar_label, str) else "")
        fig.colorbar(plt.cm.ScalarMappable(cmap=scale[0], norm=plt.Normalize(scale[1], scale[2])),
                     ax=list(axes[rows_using].ravel()), fraction=0.02, pad=0.02).set_label(text, fontsize=9)
    release_draws(fig)
    fig.suptitle(f"{title} -- by time ({BINNING_TITLES[binning]})", fontsize=14, weight="bold")

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{filename}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if record is not None:
            # "time_bins_x_cells": one row per group/row x time bin x grid cell
            source_data.record(record)
            source_data.flush("time_bins_x_cells", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def values_record(values: dict, binning: str) -> pd.DataFrame:
    """The plotted values as a long table (row, bin, cell, value) for the datasheet."""
    _, titles = BINNINGS[binning]
    rows = [{"row": key, "time_bin": titles[b - 1], "cell": cell, "value": v}
            for (key, b), s in values.items() for cell, v in s.items()]
    return pd.DataFrame(rows)


def plot_all_binned_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    kinematics: pd.DataFrame,
    labeled: pd.DataFrame,
    comparisons: list,
    genotype_sets: dict,
    binnings: tuple = ("pct_bins", "frame_bins"),
    min_frames: int = 10,
    heatmap_vmin: float = 0.3,
    grid_filename: str = "Session0001_grid.shp",
    show_first: bool = True,
    zones: dict | None = None,
    zone_sets: dict | None = None,
    zone_set_directions: dict | None = None,
) -> None:
    """Every level 1 heatmap (1, 1b-1f, 5f, 6a, 6b, 7, 7b) by time bin, for each
    comparison and binning, saved into each step's own figure folder as
    <stem>_<comparison>_<binning>.pdf (the report's Time bins selector).
    `kinematics` from add_nose_speeds(), `labeled` from label_bout_directions();
    bins are set on df_hmm's frames and matched onto both by session + s_no.
    Per-mouse maps (6a, 6b, 7, 7b) need >= min_frames frames per mouse per cell
    and bin (lower than the all-frames maps' 20, as a bin has ~1/5 of the frames);
    5f needs >= 3 passes (5 on the all-frames map).
    zone_set_directions {zone set: [TRAVEL directions]} (names in `zone_sets`,
    zones in `zones`) adds 6a/6b pooled by zone, as
    direction_heatmap_<metric>_<direction>_<zone set>_<comparison>_<binning>.pdf."""
    from compass_labyrinth.utils import build_genotype_groups, group_sample_sizes
    from compass_labyrinth.post_hoc_analysis.level_1 import direction_analysis as da
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import difference_pairs

    L1 = "figures/level_1"
    bins = add_time_bins(df_hmm[["session", "s_no"]])
    df_hmm = df_hmm.merge(bins, on=["session", "s_no"], how="left")
    kinematics = kinematics.drop(columns=["pct_bin", "frame_bin"], errors="ignore").merge(
        bins, on=["session", "s_no"], how="left")
    labeled = labeled.drop(columns=["pct_bin", "frame_bin"], errors="ignore").merge(
        bins, on=["session", "s_no"], how="left")
    pooled = [  # (folder, stem, value column or state, label, cmap, vmin, vmax, fmt, frames table)
        ("plot1_heatmap", "all_genotypes_grid_heatmap", "state", "Slow / Surveillance proportion", "YlOrRd", heatmap_vmin, 1.0, "{:.2f}", "hmm"),
        ("plot1b_velocity_heatmap", "all_genotypes_velocity_heatmap", "velocity", "Mean velocity (belly)", "YlGnBu", 0, 300, "{:.0f}", "hmm"),
        ("plot1c_sternum_speed", "all_genotypes_sternum_speed", "sternum_speed", "Sternum speed", "YlGnBu", 0, 300, "{:.0f}", "kin"),
        ("plot1d_nose_speed_allocentric", "all_genotypes_nose_speed_allocentric", "nose_speed_allo", "Allocentric nose speed", "YlGnBu", 0, 400, "{:.0f}", "kin"),
        ("plot1e_nose_speed_egocentric", "all_genotypes_nose_speed_egocentric", "nose_speed_ego", "Egocentric nose speed", "YlGnBu", 0, 300, "{:.0f}", "kin"),
        ("plot1f_head_angle_absolute", "all_genotypes_head_angle_absolute", "head_angle_abs", "Absolute head angle (deg)", "magma_r", 0, 60, "{:.0f}", "kin"),
    ]
    first = True
    for gs in comparisons:
        hmm_g = build_genotype_groups(df_hmm, gs, genotype_sets)
        kin_g = build_genotype_groups(kinematics, gs, genotype_sets)
        lab_g = build_genotype_groups(labeled, gs, genotype_sets)
        groups = [g for g in genotype_sets[gs] if g in set(hmm_g["genotype"])]
        n = group_sample_sizes(hmm_g, "genotype")
        group_rows = [(g, g) for g in groups]
        pairs = difference_pairs(groups)

        def diff_rows(prefixes=((None, ""),)):
            # B - A rows; prefix p turns row keys g into (p, g) (6a-6c), label prefix e.g. "Outbound · "
            key = lambda p, g: g if p is None else (p, g)
            return [(("diff", p, b, a), f"{lab}{b} − {a}", key(p, b), key(p, a))
                    for p, lab in prefixes for b, a in pairs]
        for binning in binnings:
            bin_col, _ = BINNINGS[binning]
            # mice with any frames in each bin
            per_bin = hmm_g.dropna(subset=[bin_col]).groupby(["genotype", bin_col])["session"].nunique()
            mice_in = {(g, int(b)): v for (g, b), v in per_bin.items()}
            group_bin_n = {(g, b): mice_in.get((g, b), 0) for g in groups for b in range(1, 6)}

            def show():
                nonlocal first
                out, first = first and show_first, False
                return out

            # 1, 1b-1f: pooled over the group's frames in the cell and bin
            for folder, stem, col, label, cmap, lo, hi, fmt, src in pooled:
                frames = hmm_g if src == "hmm" else kin_g

                def fn(fr, col=col):
                    v = (fr["hmm_state"] == 1).astype(float) if col == "state" else fr[col]
                    return fr.assign(value=v).dropna(subset=["value"]).groupby(
                        ["genotype", "grid_number"])["value"].mean().reset_index()

                vals = binned_cell_values(frames, bin_col, groups, fn)
                plot_binned_heatmaps(config, vals, group_rows, binning, cmap, lo, hi, fmt, label, label,
                                     grid_filename, row_n=n, bin_n=group_bin_n, diff_rows=diff_rows(),
                                     record=values_record(vals, binning),
                                     filename=f"{stem}_{gs}_{binning}", figures_dirname=f"{L1}/{folder}",
                                     show_fig=show())

            # 6a / 6b: one figure per travel direction, per mouse
            bins_present = sorted(int(b) for b in lab_g[bin_col].dropna().unique())
            cells_by_bin = {b: da.cell_phase_table(lab_g[lab_g[bin_col] == b]) for b in bins_present}
            travel_by_bin = {b: da.cell_phase_table(da.with_travel(lab_g[lab_g[bin_col] == b])) for b in bins_present}
            zone_tables = {zs: {b: da.zone_direction_table(c, zones, zone_sets[zs]) for b, c in travel_by_bin.items()}
                           for zs in (zone_set_directions or {})}
            for metric, folder, cmap, lo, hi, fmt, label in [
                    ("surveillance", "plot6a_direction_heatmap_surveillance", "YlOrRd", heatmap_vmin, 1.0, "{:.2f}", "Slow / Surveillance share"),
                    ("velocity", "plot6b_direction_heatmap_velocity", "YlGnBu", 0, 300, "{:.0f}", "Belly speed")]:
                # pooled by zone: each zone's value painted on its cells (cells outside the set grey)
                for zs, directions in (zone_set_directions or {}).items():
                    for direction in directions:
                        vals = {}
                        for b, zt in zone_tables[zs].items():
                            m = zt[(zt["phase"] == direction) & (zt["n"] >= min_frames)].groupby(["genotype", "zone"])[metric].mean()
                            for g in groups:
                                vals[(g, b)] = pd.Series({cell: m.get((g, z), np.nan) for z in zone_sets[zs]
                                                          for cell in zones[z]}, dtype=float).dropna()
                        plot_binned_heatmaps(config, vals, group_rows, binning, cmap, lo, hi, fmt,
                                             f"{label} -- {da.TRAVEL[direction]}, by zone ({zs})", label, grid_filename,
                                             row_n=n, bin_n=group_bin_n, diff_rows=diff_rows(),
                                             record=values_record(vals, binning),
                                             filename=f"direction_heatmap_{metric}_{direction}_{zs}_{gs}_{binning}",
                                             figures_dirname=f"{L1}/{folder}", show_fig=False)
                v_by_bin = {b: da.direction_cell_values(c, metric, min_frames=min_frames) for b, c in travel_by_bin.items()}
                for direction, title in da.TRAVEL.items():
                    vals = {}
                    for b, v in v_by_bin.items():
                        for g in groups:
                            sub = v[v["genotype"] == g].set_index("grid_number")
                            vals[(g, b)] = sub[direction] if direction in sub else pd.Series(dtype=float)
                    plot_binned_heatmaps(config, vals, group_rows, binning, cmap, lo, hi, fmt, f"{label} -- {title}",
                                         label, grid_filename, row_n=n, bin_n=group_bin_n, diff_rows=diff_rows(),
                                         record=values_record(vals, binning),
                                         filename=f"direction_heatmap_{metric}_{direction}_{gs}_{binning}",
                                         figures_dirname=f"{L1}/{folder}", show_fig=show())

            # 5f: turn-back and turn-in probability rows, per mouse (>= 3 passes per cell and bin)
            vals = {}
            for b in bins_present:
                v = da.reversal_cell_values(da.cell_turn_table(lab_g[lab_g[bin_col] == b]), min_passes=3)
                for col in ("p_backtrack", "p_reentry"):
                    for g in groups:
                        vals[((col, g), b)] = v[v["genotype"] == g].set_index("grid_number")[col]
            rows = [(("p_backtrack", g), f"Backtrack \u00b7 {g}", "Oranges", 0, 0.2) for g in groups] + \
                   [(("p_reentry", g), f"Re-entry \u00b7 {g}", "Oranges", 0, 0.2) for g in groups]
            plot_binned_heatmaps(config, vals, rows, binning, "Oranges", 0, 0.2, "{:.2f}",
                                 "How often a pass through the cell turns around",
                                 "P(turn back) / P(turn in again) per pass", grid_filename,
                                 row_n={(c, g): n.get(g) for c in ("p_backtrack", "p_reentry") for g in groups},
                                 bin_n={((c, g), b): v for (g, b), v in group_bin_n.items() for c in ("p_backtrack", "p_reentry")},
                                 diff_rows=diff_rows([("p_backtrack", "Backtrack · "), ("p_reentry", "Re-entry · ")]),
                                 record=values_record(vals, binning), filename=f"reversal_heatmap_{gs}_{binning}",
                                 figures_dirname=f"{L1}/plot5f_turn_heatmap", show_fig=show())

            # 7 / 7b: signed head angle (and its absolute value) per direction, with arrows
            angles_by_bin = {b: da.cell_phase_angle_table(da.with_travel(lab_g[lab_g[bin_col] == b]), kinematics)
                             for b in cells_by_bin}
            for phase in da.TRAVEL:
                t_by_bin = {b: a[(a["phase"] == phase) & (a["n"] >= min_frames)] for b, a in angles_by_bin.items()}
                for absolute, folder, stem in [(False, "plot7_head_angle_by_direction", "head_angle"),
                                               (True, "plot7b_head_angle_abs_by_direction", "head_angle_abs")]:
                    cmap, lo, hi = ("magma_r", 0, 30) if absolute else ("RdBu_r", -30, 30)
                    vals = {}
                    for b, t in t_by_bin.items():
                        m = t.groupby(["genotype", "grid_number"])["head_angle"].mean()
                        for g in groups:
                            s_ = m.xs(g, level="genotype") if g in m.index.get_level_values(0) else pd.Series(dtype=float)
                            vals[(g, b)] = s_.abs() if absolute else s_

                    def arrows(ax, g, b, absolute=absolute, cmap=cmap, lo=lo, hi=hi, phase=phase):
                        t = t_by_bin.get(b)
                        if t is not None and len(t):
                            da._draw_head_arrows([(ax, g)], config, grid_filename, t, phase, absolute, cmap, lo, hi)

                    label = "|Signed head angle| (deg)" if absolute else "Signed head angle (deg), + = right"
                    plot_binned_heatmaps(config, vals, group_rows, binning, cmap, lo, hi, "{:.0f}",
                                         f"{'|Head angle|' if absolute else 'Head angle'} -- {da.PHASE_TITLES[phase]}",
                                         label, grid_filename, row_n=n, bin_n=group_bin_n, arrows_fn=arrows,
                                         diff_rows=diff_rows(),
                                         record=values_record(vals, binning),
                                         filename=f"{stem}_{phase}_{gs}_{binning}",
                                         figures_dirname=f"{L1}/{folder}", show_fig=show())
