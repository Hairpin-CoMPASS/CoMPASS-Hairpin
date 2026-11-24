"""
HEATMAP REPRESENTATIONS
Author: Shreya Bangera
Goal:
    ├── Grid-wise probability heatmaps showing the proportion of time spent in a specific HMM state across all sessions.
    ├── Interactive version of the visualization available too.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING

import compass_labyrinth.source_data as source_data
from compass_labyrinth.utils import genotype_order as _config_genotype_order


warnings.simplefilter(action="ignore", category=FutureWarning)


# hmm_state=0 is not part of the fitted HMM itself -- it's frames excluded before
# fitting and re-attached afterward (see notebook 04 / merge_stationary_frames()).
HMM_STATE_LABELS = {
    0: "Stationary",
    1: "Slow / Surveillance",
    2: "Fast / Ambulatory",
}


def build_warm_to_cold_cmap(n: int = 256) -> ListedColormap:
    """
    Build a low=warm(red), high=cold(blue) colormap that stays vividly
    saturated throughout -- standard diverging colormaps ("RdBu", "RdYlBu",
    "coolwarm") all fade to a pale/near-white (or, if interpolated in raw
    RGB, muddy gray-brown) midpoint, since they're designed to mark a
    meaningful "zero"/baseline value there. That's not wanted for a plain
    sequential warm-to-cold gradient with no special midpoint.

    Instead of interpolating color channels directly, this sweeps HUE from
    red (0deg) to blue (240deg) the "short way" around the color wheel --
    through magenta/violet (300deg -> 240deg) rather than the "long way"
    through orange/yellow/green (0deg -> 240deg increasing) -- while holding
    saturation roughly constant.

    VALUE (brightness) dips at both ends and peaks in the middle: dark red ->
    bright magenta -> dark blue, rather than a flat/monotonic ramp -- so
    neither endpoint reads as washed-out next to the middle.

    Parameters
    -----------
    n : int
        Number of discrete color steps.

    Returns
    --------
    ListedColormap
        Pass directly as `cmap=` to plot_grid_heatmap()/plot_all_genotype_heatmaps()
        (or any matplotlib plotting call) -- no need to register it by name.
    """
    t = np.linspace(0, 1, n)
    hue_deg = (0 - 120 * t) % 360  # 0 (red) -> 300 (magenta) -> 240 (blue)
    hue = hue_deg / 360.0
    sat = np.full(n, 0.85)
    val = 0.55 + 0.35 * np.sin(np.pi * t)  # darker at both ends, brighter in the middle
    rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=1))
    return ListedColormap(rgb, name="warm_cold")


def build_yellow_to_purple_cmap(n: int = 256) -> ListedColormap:
    """
    Same no-white-midpoint hue-sweep design as build_warm_to_cold_cmap(), but
    low=yellow, high=purple. Sweeps HUE from yellow (50deg) down through
    orange/red/magenta to purple (280deg) -- the short way around the color
    wheel (130deg of travel) -- rather than the long way through green/cyan/
    blue (230deg of travel), so the whole gradient stays vividly saturated
    with no washed-out point.

    Unlike build_warm_to_cold_cmap()'s monotonic value ramp, VALUE (brightness)
    here dips at both ends and peaks in the middle: yellow and purple read
    lighter than red/magenta at the same value/saturation just from hue alone,
    so a flat or monotonic ramp made one end look artificially pale next to
    the middle. Darkening both ends and brightening the middle balances that
    out -- dark gold -> bright red/magenta -> dark violet.

    Parameters
    -----------
    n : int
        Number of discrete color steps.

    Returns
    --------
    ListedColormap
        Pass directly as `cmap=` to plot_grid_heatmap()/plot_all_genotype_heatmaps()
        (or any matplotlib plotting call) -- no need to register it by name.
    """
    t = np.linspace(0, 1, n)
    hue_deg = (50 - 130 * t) % 360  # 50 (yellow) -> 0 (red) -> 280 (purple)
    hue = hue_deg / 360.0
    sat = np.full(n, 0.85)
    val = 0.55 + 0.35 * np.sin(np.pi * t)  # darker at both ends, brighter in the middle
    rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=1))
    return ListedColormap(rgb, name="yellow_purple")


def annotate_grid_values(
    ax: plt.Axes,
    grid: gpd.GeoDataFrame,
    column: str = "state1_proportion",
    fontsize: float = 6,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Label each grid cell with its value, so cells with similar colors (e.g.
    several near 1.0) remain distinguishable by their exact number.

    Text color switches from black to white above the midpoint of [vmin, vmax]
    (matching the colormap's own vmin/vmax) to stay readable against both the
    light and dark end of a sequential colormap.
    """
    midpoint = (vmin + vmax) / 2
    for _, row in grid.iterrows():
        value = row[column]
        if pd.isna(value):
            continue
        centroid = row.geometry.centroid
        text_color = "white" if value > midpoint else "black"
        ax.annotate(
            f"{value:.2f}",
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
        )


##################################################################
# Plot 1: Heatmap Representations of HMM States
###################################################################
def compute_state_proportion(
    df: pd.DataFrame,
    genotype_name: str,
    hmm_state: int = 2,
) -> pd.DataFrame:
    """
    Compute state proportions and return filtered dataframe for given genotype and HMM state.

    Parameters
    -----------
    df : pd.DataFrame
        DataFrame containing columns: 'genotype', 'grid_number', 'hmm_state', 'x', 'y'.
    genotype_name : str
        Name of the genotype to filter.
    hmm_state : int
        HMM state to filter.

    Returns
    --------
    pd.DataFrame
        DataFrame with computed proportions for the specified genotype and HMM state.
    """
    st_cnt = df.groupby(["genotype", "grid_number", "hmm_state"]).size().rename("cnt").reset_index()
    gn_cnt = df.groupby(["genotype", "grid_number"]).size().rename("tot").reset_index()
    x_y = df.groupby(["genotype", "grid_number"]).agg({"x": "mean", "y": "mean"}).reset_index()
    state_count = st_cnt.merge(gn_cnt, on=["genotype", "grid_number"], how="left")
    state_count["prop"] = state_count["cnt"] / state_count["tot"]
    state_count = state_count.merge(x_y, on=["genotype", "grid_number"], how="left")

    return state_count[(state_count.genotype == genotype_name) & (state_count["hmm_state"] == hmm_state)].reset_index(
        drop=True
    )


def compute_mean_velocity(
    df: pd.DataFrame,
    genotype_name: str,
) -> pd.DataFrame:
    """
    Compute mean velocity per grid cell and return filtered dataframe for the given genotype.

    Same output shape as compute_state_proportion() (grid_number, x, y, plus
    the value column), so it drops into the same
    map_points_to_grid()/merge_state_proportions_to_grid()/plot_grid_heatmap()
    pipeline -- just pass value_col="velocity" to merge_state_proportions_to_grid()
    and value_column="mean_velocity" (or whatever target_col you merged it as)
    to plot_grid_heatmap()/plot_all_genotype_heatmaps().

    Parameters
    -----------
    df : pd.DataFrame
        DataFrame containing columns: 'genotype', 'grid_number', 'velocity', 'x', 'y'.
    genotype_name : str
        Name of the genotype to filter.

    Returns
    --------
    pd.DataFrame
        DataFrame with mean velocity per grid cell for the specified genotype.
    """
    vel_by_grid = df.groupby(["genotype", "grid_number"])["velocity"].mean().rename("velocity").reset_index()
    x_y = df.groupby(["genotype", "grid_number"]).agg({"x": "mean", "y": "mean"}).reset_index()
    vel_by_grid = vel_by_grid.merge(x_y, on=["genotype", "grid_number"], how="left")

    return vel_by_grid[vel_by_grid.genotype == genotype_name].reset_index(drop=True)


def create_grid_geodata(config: dict, grid_filename: str) -> gpd.GeoDataFrame:
    """Load the shapefile grid as GeoDataFrame."""
    gridfile = Path(config["project_path_full"]) / "grid_files" / grid_filename
    return gpd.read_file(gridfile)


def map_points_to_grid(df_points: pd.DataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map mean x, y points to the grid."""
    points = [Point(xy) for xy in zip(df_points["x"], df_points["y"])]
    pnt_gpd = gpd.GeoDataFrame(geometry=points, index=np.arange(len(points)), crs=grid.crs)
    pointInPolys = gpd.tools.sjoin(pnt_gpd, grid, predicate="within", how="left")
    return pointInPolys


def merge_state_proportions_to_grid(
    grid: gpd.GeoDataFrame,
    df_props: pd.DataFrame,
    value_col: str = "prop",
    target_col: str = "state1_proportion",
) -> gpd.GeoDataFrame:
    """Merge a per-grid-cell value (state proportion, mean velocity, etc.) to shapefile grid polygons.

    value_col : str
        Column in df_props holding the value to merge -- "prop" (default)
        from compute_state_proportion(), or "velocity" from compute_mean_velocity().
    target_col : str
        Column name the merged value ends up under on the returned grid.
        plot_grid_heatmap()/plot_all_genotype_heatmaps() read this column via
        their own value_column parameter (default "state1_proportion" to
        match here) -- pass the same name to both this and the plot call.
    """
    prop_by_grid = df_props[["grid_number", value_col]].copy()
    prop_by_grid = prop_by_grid.rename(columns={value_col: target_col})
    return grid.merge(
        prop_by_grid,
        left_on="FID",
        right_on="grid_number",
        how="left",
    )


def plot_grid_heatmap(
    config: dict,
    grid: gpd.GeoDataFrame,
    genotype_name: str,
    hmm_state: int | None = None,
    highlight_grids: str | None = None,
    target_grids: str | None = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    value_column: str = "state1_proportion",
    metric_label: str | None = None,
    filename_stem: str = "grid_heatmap",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot grid heatmap for state proportions (or any other per-grid-cell value).

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    grid : gpd.GeoDataFrame
        GeoDataFrame of the grid with state proportions (or another value,
        see value_column) merged.
    genotype_name : str
        Name of the genotype to be plotted.
    hmm_state : int | None
        The hmm_state this heatmap's proportions were computed for (used only to
        make the title explicit, e.g. "Fast / Ambulatory" instead of a bare
        genotype name -- doesn't affect the plotted data itself). Ignored
        (title falls back to metric_label) when metric_label is given.
    highlight_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    cmap : str
        Colormap to use for the heatmap.
    vmin : float
        Lower bound of the color scale (and the value annotations' text-color
        threshold). Narrow this above 0 to spread out contrast among values
        clustered near vmax. HMM state proportions are always in [0, 1]; a
        different metric (e.g. velocity) will need its own vmin/vmax.
    vmax : float
        Upper bound of the color scale.
    value_column : str
        Column on `grid` to plot -- "state1_proportion" (default, from
        merge_state_proportions_to_grid()'s own default target_col) for HMM
        state proportions, or whatever target_col you merged another value
        (e.g. mean velocity) under.
    metric_label : str | None
        Overrides the hmm_state-based title wording entirely, e.g. "Mean
        Velocity" -- use this for any value_column that isn't an HMM state
        proportion, since "Fast / Ambulatory Proportion" wouldn't make sense
        for a velocity heatmap.
    filename_stem : str
        Base name used in the saved filename, "{genotype_name}_{filename_stem}
        {filename_suffix}.png" -- override for a non-state-proportion
        value_column (e.g. "velocity_heatmap") so the saved file is
        identifiable on its own, without relying on figures_dirname.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : plt.Figure | None
        Matplotlib Figure object if return_fig is True, else None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    grid.plot(
        column=value_column,
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        edgecolor="0.8",
        legend=True,
        cax=cax,
        alpha=1.0,
    )
    annotate_grid_values(ax, grid, column=value_column, vmin=vmin, vmax=vmax)

    if highlight_grids is not None:
        highlight_grids_values = NODE_TYPE_MAPPING[highlight_grids]
        edge_subset = grid[grid["FID"].isin(highlight_grids_values)]
        edge_subset.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2)
    if target_grids is not None:
        target_grids_values = NODE_TYPE_MAPPING[target_grids]
        target_nodes = grid[grid["FID"].isin(target_grids_values)]
        # Dissolve into one polygon before plotting -- plotting the individual
        # cells draws a border around each one (looks like a thick grid of
        # separate squares); dissolving first outlines only the union's
        # outer boundary, i.e. one square.
        target_union = gpd.GeoDataFrame(geometry=[target_nodes.union_all()], crs=grid.crs)
        target_union.plot(ax=ax, edgecolor="yellow", facecolor="none", linewidth=1.5)

    # Flip the y-axis to match video coordinates (row 0 is the top of the maze,
    # but the shapefile's y increases downward like raw pixel/video coordinates)
    ax.invert_yaxis()

    if metric_label:
        title = f"{genotype_name} - {metric_label}"
    else:
        state_label = HMM_STATE_LABELS.get(hmm_state, f"State {hmm_state}") if hmm_state is not None else None
        title = f"{genotype_name} - {state_label} Proportion" if state_label else f"{genotype_name} (Belly Tracking)"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{genotype_name}_{filename_stem}{filename_suffix}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close()  # not shown -> don't let Jupyter display it at cell end

    # Return figure
    if return_fig:
        return fig



##################################################################
# Genotype panels + difference panels (shared layout)
##################################################################
DIFF_CMAP = "RdBu_r"  # difference panels: blue = lower in the second group, white ~ 0, red = higher


def hold_draws(fig) -> None:
    """Stop geopandas' .plot() from redrawing the whole figure after every call
    (it calls canvas.draw_idle(), which non-interactive backends run at once):
    building a many-panel figure then costs one draw instead of one per panel."""
    fig.canvas.draw_idle = lambda *args, **kwargs: None


def release_draws(fig) -> None:
    """Undo hold_draws() once the figure is built."""
    fig.canvas.__dict__.pop("draw_idle", None)


def difference_pairs(genotypes: list) -> list:
    """(second, first) group pairs for difference panels: 2 groups -> B - A;
    4 groups (E3, E3NLF, E4, E4NLF order) -> E3NLF - E3, E4NLF - E4,
    E4 - E3, E4NLF - E3NLF; any other count -> none."""
    g = list(genotypes)
    if len(g) == 2:
        return [(g[1], g[0])]
    if len(g) == 4:
        return [(g[1], g[0]), (g[3], g[2]), (g[2], g[0]), (g[3], g[1])]
    return []


def genotype_panel_axes(genotypes: list, one_row: bool = False, differences: bool = True,
                        panel_w: float = 6, panel_h: float = 5):
    """A figure with one panel per genotype group plus difference panels
    (second group minus first):
      2 groups: A, B on top, B - A centered underneath;
      4 groups as 2x2 (one_row=False, e.g. E3, E3NLF / E4, E4NLF): a 3x3 grid
        without its bottom-right square -- each row's difference on its right
        (E3NLF - E3, E4NLF - E4), each column's underneath (E4 - E3, E4NLF - E3NLF);
      4 groups in one row (one_row=True): a second row with the same 4
        differences;
      anything else: the genotype panels only.
    Returns (fig, [(ax, ("group", g)) or (ax, ("diff", b, a))])."""
    from matplotlib.gridspec import GridSpec

    n = len(genotypes)
    g = genotypes
    if differences and n == 2:
        fig = plt.figure(figsize=(2 * panel_w, 2 * panel_h))
        gs = GridSpec(2, 4, figure=fig)
        slots = [(gs[0, 0:2], ("group", g[0])), (gs[0, 2:4], ("group", g[1])), (gs[1, 1:3], ("diff", g[1], g[0]))]
    elif differences and n == 4 and not one_row:
        fig = plt.figure(figsize=(3 * panel_w, 3 * panel_h))
        gs = GridSpec(3, 3, figure=fig)
        slots = [(gs[0, 0], ("group", g[0])), (gs[0, 1], ("group", g[1])), (gs[0, 2], ("diff", g[1], g[0])),
                 (gs[1, 0], ("group", g[2])), (gs[1, 1], ("group", g[3])), (gs[1, 2], ("diff", g[3], g[2])),
                 (gs[2, 0], ("diff", g[2], g[0])), (gs[2, 1], ("diff", g[3], g[1]))]
    elif differences and n == 4 and one_row:
        fig = plt.figure(figsize=(4 * panel_w, 2 * panel_h))
        gs = GridSpec(2, 4, figure=fig)
        slots = [(gs[0, i], ("group", g[i])) for i in range(4)]
        slots += [(gs[1, i], ("diff", b, a)) for i, (b, a) in enumerate(difference_pairs(g))]
    else:
        n_cols = n if one_row else math.ceil(n ** 0.5)
        n_rows = 1 if one_row else math.ceil(n / n_cols)
        fig = plt.figure(figsize=(panel_w * n_cols, panel_h * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        slots = [(gs[i // n_cols, i % n_cols], ("group", gg)) for i, gg in enumerate(g)]
    hold_draws(fig)  # callers release_draws(fig) once all panels are drawn
    return fig, [(fig.add_subplot(slot), spec) for slot, spec in slots]


def robust_diff_lim(diffs, q: float = 95) -> float:
    """Symmetric color limit for difference panels: the q-th percentile of
    |difference| over all cells/zones of all difference panels, rounded up to
    a readable number."""
    v = np.abs(np.concatenate([np.asarray(d, dtype=float).ravel() for d in diffs])) if len(diffs) else np.array([])
    v = v[np.isfinite(v)]
    if not len(v):
        return 1.0
    lim = float(np.percentile(v, q))
    if lim <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(lim))
    for step in (1, 2, 2.5, 5, 10):
        if lim <= step * mag:
            return step * mag
    return lim


def annotate_diverging(ax, gdf: gpd.GeoDataFrame, column: str, lim: float, fmt: str, fontsize: float = 5) -> None:
    """Value labels for a two-sided (difference) panel: white on the strongly
    colored ends, black near zero."""
    for _, row in gdf.iterrows():
        v = row[column]
        if pd.isna(v):
            continue
        pt = row.geometry.centroid
        ax.annotate(fmt.format(v), xy=(pt.x, pt.y), ha="center", va="center", fontsize=fontsize,
                    color="white" if abs(v) > 0.55 * lim else "black")


##################################################################
# Heatmap Representations of HMM States for all genotypes
###################################################################
def plot_all_genotype_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    grid_filename: str,
    highlight_grids: str | None = None,
    target_grids: str | None = None,
    hmm_state: int = 1,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    compute_fn=None,
    value_col: str = "prop",
    value_column: str = "state1_proportion",
    metric_label: str | None = None,
    filename_stem: str = "grid_heatmap",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    genotype_order: list | None = None,
    tracking_label: str = "Belly Tracking",
    show_differences: bool = True,
    diff_lim: float | None = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot grid heatmaps for all genotypes.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'genotype' and 'grid_number' columns.
    grid_filename : str
        Filename of the grid shapefile.
    highlight_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    hmm_state : int
        HMM state to filter. Ignored if compute_fn is given.
    cmap : str
        Colormap to use for the heatmap.
    vmin : float
        Lower bound of the color scale (and the value annotations' text-color
        threshold). Narrow this above 0 to spread out contrast among values
        clustered near vmax. HMM state proportions are always in [0, 1]; a
        different metric (e.g. velocity) will need its own vmin/vmax.
    vmax : float
        Upper bound of the color scale.
    compute_fn : callable | None
        (df_hmm, genotype) -> DataFrame, same output shape as
        compute_state_proportion(). Defaults to
        `lambda df, g: compute_state_proportion(df, g, hmm_state)`. Pass
        `lambda df, g: compute_mean_velocity(df, g)` (together with
        value_col="velocity", value_column="mean_velocity", metric_label=
        "Mean Velocity") for a velocity heatmap instead of an HMM state proportion one.
    value_col : str
        Column in compute_fn's output holding the value to merge -- forwarded
        to merge_state_proportions_to_grid()'s value_col.
    value_column : str
        Column name the merged value is plotted from -- forwarded to
        merge_state_proportions_to_grid()'s target_col.
    metric_label : str | None
        Overrides the hmm_state-based suptitle wording entirely, e.g. "Mean
        Velocity" -- use whenever compute_fn isn't an HMM state proportion.
    filename_stem : str
        Base name used in the saved filename, "all_genotypes_{filename_stem}
        {filename_suffix}.pdf" -- override for a non-state-proportion
        compute_fn (e.g. "velocity_heatmap") so the saved file is
        identifiable on its own, without relying on figures_dirname.
    genotype_order : list | None
        Panel order, left to right then down (e.g. list(GENOTYPE_SETS["wt_nlf"])
        for WT before NLF). Genotypes not listed follow alphabetically;
        None sorts all alphabetically.
    tracking_label : str
        The body part named in each panel title, e.g. "Nose".
    show_differences : bool
        Add difference panels, second group minus first (red = higher in the
        second group); layout in genotype_panel_axes().
    diff_lim : float | None
        Color limit (+-) for the difference panels; None = the 95th percentile
        of |difference| over their cells, rounded.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : plt.Figure | None
        Matplotlib Figure object if return_fig is True, else None.
    """
    if compute_fn is None:
        compute_fn = lambda df, g: compute_state_proportion(df, g, hmm_state)  # noqa: E731

    present = set(df_hmm["genotype"].unique())
    if genotype_order is not None:
        genotypes = [g for g in genotype_order if g in present] + sorted(present - set(genotype_order))
    else:
        genotypes = _config_genotype_order(present)

    grid = create_grid_geodata(config, grid_filename)

    # Per-genotype per-cell values first, so the difference panels can use them
    _recorded_by_genotype, mapped = [], {}
    for genotype in genotypes:
        # Only this genotype's rows -- compute_fn filters to it anyway
        state_df = compute_fn(df_hmm[df_hmm["genotype"] == genotype], genotype)
        _recorded_by_genotype.append(state_df.assign(genotype=genotype))
        mapped[genotype] = merge_state_proportions_to_grid(grid, state_df, value_col=value_col, target_col=value_column)

    fig, panels = genotype_panel_axes(genotypes, one_row=False, differences=show_differences)
    diff_maps = {}
    for _, spec in panels:
        if spec[0] == "diff":
            b, a = spec[1], spec[2]
            vb = mapped[b].drop_duplicates("FID").set_index("FID")[value_column]
            va = mapped[a].drop_duplicates("FID").set_index("FID")[value_column]
            diff_maps[(b, a)] = grid.assign(difference=grid["FID"].map(vb - va))
    lim = diff_lim if diff_lim is not None else robust_diff_lim([d["difference"] for d in diff_maps.values()])
    fmt = "{:+.2f}" if value_col == "prop" else "{:+.0f}"

    for ax, spec in panels:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if spec[0] == "group":
            genotype = spec[1]
            grid_mapped = mapped[genotype]
            grid_mapped.plot(column=value_column, cmap=cmap, linewidth=0.8, ax=ax, vmin=vmin, vmax=vmax,
                             edgecolor="0.8", legend=True, cax=cax, alpha=1.0)
            annotate_grid_values(ax, grid_mapped, column=value_column, fontsize=5, vmin=vmin, vmax=vmax)
            n_sessions = df_hmm.loc[df_hmm["genotype"] == genotype, "session"].nunique()
            ax.set_title(f"{genotype} ({tracking_label}, n={n_sessions})", fontsize=12)
        else:
            b, a = spec[1], spec[2]
            dm = diff_maps[(b, a)]
            dm.plot(column="difference", cmap=DIFF_CMAP, linewidth=0.8, ax=ax, vmin=-lim, vmax=lim,
                    edgecolor="0.8", legend=True, cax=cax, missing_kwds={"color": "0.9"})
            annotate_diverging(ax, dm, "difference", lim, fmt)
            ax.set_title(f"{b} \u2212 {a}", fontsize=12, weight="bold")

        if highlight_grids is not None:
            edge_subset = grid[grid["FID"].isin(NODE_TYPE_MAPPING[highlight_grids])]
            edge_subset.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2)
        if target_grids is not None:
            target_subset = grid[grid["FID"].isin(NODE_TYPE_MAPPING[target_grids])]
            # Dissolve into one polygon so only the union's outer boundary is
            # outlined (one square), not a border around each individual cell.
            target_union = gpd.GeoDataFrame(geometry=[target_subset.union_all()], crs=grid.crs)
            target_union.plot(ax=ax, edgecolor="yellow", facecolor="none", linewidth=1.5)

        # Flip the y-axis to match video coordinates (see plot_grid_heatmap)
        ax.invert_yaxis()
        ax.axis("off")

    release_draws(fig)
    if metric_label:
        suptitle = f"{metric_label} Across Genotypes"
    else:
        state_label = HMM_STATE_LABELS.get(hmm_state, f"State {hmm_state}")
        suptitle = f"{state_label} Proportion Across Genotypes"
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"all_genotypes_{filename_stem}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "grid_cells" -- one row per genotype per grid cell (not a
        # session-level table; state proportion and velocity are already
        # pooled across every session of that genotype) -- shared by both
        # Plot 1 (state proportion) and Plot 1b (velocity) instead of each
        # getting its own narrow sheet.
        # "genotype_group", not "genotype": source_data drops a bare
        # "genotype" column (it re-joins genotype from the mouse metadata by
        # session), and these pooled per-cell rows have no session to join.
        source_data.record(
            pd.concat(_recorded_by_genotype, ignore_index=True).rename(columns={"genotype": "genotype_group"})
        )
        source_data.flush("grid_cells", figure=save_path.stem)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close()  # not shown -> don't let Jupyter display it at cell end

    # Return figure
    if return_fig:
        return fig


##################################################################
# Plot 1: Heatmap Representations of HMM States - INTERACTIVE MAP
###################################################################
def get_grid_centroids(grid):
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    return {row["FID"]: (row["x"], row["y"]) for _, row in grid.iterrows()}


def overlay_trajectory_lines_plotly(
    fig: go.Figure,
    df_hmm: pd.DataFrame,
    genotype_name: str,
    grid_centroids: dict,
    top_percent: float = 0.1,
):
    """
    Overlay trajectory lines on the interactive heatmap.

    Parameters
    -----------
    fig : go.Figure
        Plotly Figure object to overlay lines on.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'genotype' and 'grid_number' columns.
    genotype_name : str
        Name of the genotype to filter.
    grid_centroids : dict
        Dictionary mapping grid numbers to their centroid coordinates.
    top_percent : float
        Top percentage of transitions to visualize.

    Returns
    --------
    None
    """
    df_geno = df_hmm[df_hmm["genotype"] == genotype_name].copy()
    df_geno["grid_next"] = df_geno["grid_number"].shift(-1)
    df_geno["grid_prev"] = df_geno["grid_number"]
    transitions = df_geno[["grid_prev", "grid_next"]].dropna()
    transitions = transitions[transitions["grid_prev"] != transitions["grid_next"]].astype(int)
    transitions["pair"] = list(zip(transitions["grid_prev"], transitions["grid_next"]))
    trans_counts = transitions["pair"].value_counts()

    top_n = int(len(trans_counts) * top_percent)
    trans_counts = trans_counts.iloc[:top_n]

    max_freq = trans_counts.max()
    min_freq = trans_counts.min()

    for (g1, g2), freq in trans_counts.items():
        if g1 in grid_centroids and g2 in grid_centroids:
            pt1, pt2 = grid_centroids[g1], grid_centroids[g2]
            if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                continue
            # Smooth interpolation
            x_vals = np.array([pt1[0], (pt1[0] + pt2[0]) / 2, pt2[0]])
            y_vals = np.array([pt1[1], (pt1[1] + pt2[1]) / 2 + 2, pt2[1]])  # add slight curve
            spline = make_interp_spline([0, 1, 2], np.column_stack([x_vals, y_vals]), k=2)
            smooth = spline(np.linspace(0, 2, 20))
            xs, ys = smooth[:, 0], smooth[:, 1]

            line_alpha = np.interp(freq, [min_freq, max_freq], [0.2, 1.0])
            line_width = np.interp(freq, [min_freq, max_freq], [1, 10])

            fig.add_trace(
                go.Scatter(
                    x=xs.tolist(),
                    y=ys.tolist(),
                    mode="lines",
                    line=dict(color="black", width=line_width),
                    opacity=line_alpha,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


# ---------------- Plotly Heatmap ---------------- #
def plot_interactive_heatmap(
    config: dict,
    grid_mapped: gpd.GeoDataFrame,
    genotype_name: str,
    decision_grids: str | None = None,
    target_grids: str | None = None,
    entry_node_id: int | None = 0,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | go.Figure:
    """
    Plot interactive heatmap for state proportions.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    grid_mapped : gpd.GeoDataFrame
        GeoDataFrame of the grid with state proportions merged.
    genotype_name : str
        Name of the genotype to be plotted.
    decision_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    entry_node_id : int
        ID of the entry node (default is 0).
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : go.Figure | None
        Plotly Figure object if return_fig is True, else None.
    """
    fig = go.Figure()

    # Add grid cells as filled polygons
    for _, row in grid_mapped.iterrows():
        if row.geometry is None or row["state1_proportion"] is None:
            continue
        x = list(row.geometry.exterior.xy[0])
        y = list(row.geometry.exterior.xy[1])
        color_val = row["state1_proportion"]
        fillcolor = sample_colorscale("RdBu", [color_val])[0]  # RdBu reversed = blue to red
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                # fillcolor=f'rgba({255 * (1 - row["state1_proportion"])},0,{255 * row["state1_proportion"]},0.7)',
                fillcolor=fillcolor,
                line=dict(width=0.5, color="gray"),
                hoverinfo="text",
                text=f"Grid: {row['FID']}<br>Proportion: {row['state1_proportion']:.2f}",
                showlegend=False,
            )
        )

    # Highlight decision nodes in black
    for FID in NODE_TYPE_MAPPING.get(decision_grids, []):
        if FID in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == FID, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=2),
                    name="Decision Node",
                    showlegend=False,
                )
            )

    # Highlight decision nodes in black
    if entry_node_id is not None:
        if entry_node_id in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == entry_node_id, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=5),
                    name="Entry Node",
                    showlegend=False,
                )
            )

    # Highlight target nodes in yellow
    for FID in NODE_TYPE_MAPPING.get(target_grids, []):
        if FID in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == FID, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=5),
                    name="Target Node",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{genotype_name}: Grid Heatmap with Trajectories",
        xaxis=dict(showgrid=False, visible=False),
        # autorange="reversed": row 0 is the top of the maze, but the shapefile's
        # y increases downward like raw pixel/video coordinates (see plot_grid_heatmap)
        yaxis=dict(scaleanchor="x", showgrid=False, visible=False, autorange="reversed"),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Add annotations above Grid Number 0 (entry) and below 70 (target zone)
    for FID, label, valign in [(0, "entry_zone", "top"), (70, "target_zone", "bottom")]:
        if FID in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == FID, "geometry"].values[0]
            centroid_x = poly.centroid.x
            centroid_y = poly.centroid.y

            offset = 55 if valign == "top" else -55

            fig.add_annotation(
                x=centroid_x,
                y=centroid_y + offset,
                text=label,
                showarrow=False,
                font=dict(color="black", size=14, family="Arial"),
                xanchor="center",
            )

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / f"{genotype_name}_interactive_grid_heatmap.html"
        fig.write_html(save_path)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        fig.show()

    # Return figure
    if return_fig:
        return fig


###################################################################
# Interactive Heatmaps for multiple genotypes
###################################################################
def plot_all_genotype_interactive_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    grid_filename: str,
    hmm_state: int = 2,
    decision_grids: str | None = None,
    target_grids: str | None = None,
    entry_node_id: int | None = 0,
    top_percent: float = 0.1,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | go.Figure:
    """
    Plot interactive grid heatmaps for all genotypes.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'genotype' and 'grid_number' columns.
    grid_filename : str
        Filename of the grid shapefile.
    hmm_state : int
        HMM state to filter.
    decision_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    entry_node_id : int | None
        ID of the entry node (default is 0).
    top_percent : float
        Top percentage of transitions to visualize.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : go.Figure | None
        Plotly Figure object if return_fig is True, else None.
    """
    genotypes = sorted(df_hmm["genotype"].unique())
    n = len(genotypes)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    grid = create_grid_geodata(config, grid_filename)
    grid_centroids = get_grid_centroids(grid)

    # Create subplot figure
    subplot_fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=genotypes,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    subplot_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if subplot_idx >= n:
                break

            genotype = genotypes[subplot_idx]
            subplot_idx += 1
            row = i + 1
            col = j + 1

            # 1. Compute state proportions
            state_df = compute_state_proportion(df_hmm, genotype, hmm_state)

            # 2. Merge into grid
            grid_mapped = merge_state_proportions_to_grid(grid, state_df)

            # 3. Generate interactive heatmap figure
            fig = plot_interactive_heatmap(
                config=config,
                grid_mapped=grid_mapped,
                decision_grids=decision_grids,
                target_grids=target_grids,
                genotype_name=genotype,
                entry_node_id=entry_node_id,
                save_fig=False,
                show_fig=False,
                return_fig=True,
            )

            # 4. Add trajectory lines
            overlay_trajectory_lines_plotly(
                fig=fig,
                df_hmm=df_hmm,
                genotype_name=genotype,
                grid_centroids=grid_centroids,
                top_percent=top_percent,
            )

            # 5. Add traces to subplot
            for trace in fig.data:
                subplot_fig.add_trace(trace, row=row, col=col)

    # Final layout improvements
    subplot_fig.update_layout(
        title_text=f"Genotype-wise Grid Heatmaps (HMM State {hmm_state})",
        height=600 * n_rows,
        width=600 * n_cols,
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=10),
        title_x=0.5,
    )
    # Only the data traces get copied into this subplot figure above, not the
    # layout from plot_interactive_heatmap() -- so the y-axis flip has to be
    # re-applied here too (see plot_grid_heatmap for why it's needed).
    subplot_fig.update_yaxes(autorange="reversed")

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_interactive_grid_heatmap.html"
        subplot_fig.write_html(save_path)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        subplot_fig.show()

    # Return figure
    if return_fig:
        return subplot_fig


def plot_all_genotype_zone_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    grid_filename: str,
    zones: dict,
    zone_names: list,
    metric: str = "state",
    hmm_state: int = 1,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
    metric_label: str | None = None,
    filename_stem: str = "grid_heatmap",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    genotype_order: list | None = None,
    tracking_label: str = "Belly Tracking",
    show_differences: bool = True,
    diff_lim: float | None = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """plot_all_genotype_heatmaps(), pooled per ZONE instead of per cell: each
    zone of `zone_names` is colored by its value over all of the genotype's
    frames there -- metric="state": share of frames in `hmm_state`;
    metric="velocity": mean velocity -- and labeled with it; cells outside the
    zone set are hatched grey. Same panel and difference-panel layout as the
    per-cell version (genotype_panel_axes())."""
    present = set(df_hmm["genotype"].unique())
    if genotype_order is not None:
        genotypes = [g for g in genotype_order if g in present] + sorted(present - set(genotype_order))
    else:
        genotypes = _config_genotype_order(present)
    grid = create_grid_geodata(config, grid_filename)
    in_set = set().union(*(zones[z] for z in zone_names))
    target = grid[grid["FID"].isin(NODE_TYPE_MAPPING.get("target_zone", [70, 71, 80, 81]))]
    target_union = gpd.GeoDataFrame(geometry=[target.union_all()], crs=grid.crs)
    fmt_val = "{:.2f}" if metric == "state" else "{:.0f}"

    values, recorded = {}, []
    for genotype in genotypes:
        d = df_hmm[df_hmm["genotype"] == genotype]
        values[genotype] = {}
        for zone in zone_names:
            zd = d[d["grid_number"].isin(zones[zone])]
            v = np.nan if zd.empty else ((zd["hmm_state"] == hmm_state).mean() if metric == "state" else zd["velocity"].mean())
            values[genotype][zone] = v
            recorded.append({"genotype_group": genotype, "zone": zone, "value": v, "frames": len(zd)})

    fig, panels = genotype_panel_axes(genotypes, one_row=False, differences=show_differences)
    diffs = {(spec[1], spec[2]): {z: values[spec[1]][z] - values[spec[2]][z] for z in zone_names}
             for _, spec in panels if spec[0] == "diff"}
    lim = diff_lim if diff_lim is not None else robust_diff_lim([list(d.values()) for d in diffs.values()])
    fmt_diff = "{:+.2f}" if metric == "state" else "{:+.0f}"

    for ax, spec in panels:
        if spec[0] == "group":
            vals, cm, norm, fmt = values[spec[1]], plt.get_cmap(cmap), plt.Normalize(vmin, vmax), fmt_val
        else:
            vals, cm, norm, fmt = diffs[(spec[1], spec[2])], plt.get_cmap(DIFF_CMAP), plt.Normalize(-lim, lim), fmt_diff
        grid[~grid["FID"].isin(in_set)].plot(ax=ax, facecolor="0.92", edgecolor="0.8", hatch="///", linewidth=0.4)
        for zone in zone_names:
            v = vals[zone]
            shape = grid[grid["FID"].isin(zones[zone])]
            shape.plot(ax=ax, color="0.8" if pd.isna(v) else cm(norm(v)), edgecolor="0.85", linewidth=0.4)
            outline = shape.union_all()
            gpd.GeoSeries([outline], crs=grid.crs).boundary.plot(ax=ax, color="0.3", linewidth=0.8)
            if pd.notna(v):
                rgba = cm(norm(v))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                for part in (list(outline.geoms) if hasattr(outline, "geoms") else [outline]):
                    pt = part.representative_point()
                    ax.annotate(fmt.format(v), (pt.x, pt.y), ha="center", va="center", fontsize=6,
                                color="white" if dark else "black")
        target_union.plot(ax=ax, edgecolor="yellow", facecolor="none", linewidth=1.5)
        ax.invert_yaxis()
        ax.axis("off")
        if spec[0] == "group":
            n_sessions = df_hmm.loc[df_hmm["genotype"] == spec[1], "session"].nunique()
            ax.set_title(f"{spec[1]} ({tracking_label}, n={n_sessions})", fontsize=12)
        else:
            ax.set_title(f"{spec[1]} \u2212 {spec[2]}", fontsize=12, weight="bold")
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=norm), cax=cax)
    release_draws(fig)
    if metric_label:
        suptitle = f"{metric_label} Across Genotypes, per zone"
    else:
        suptitle = f"{HMM_STATE_LABELS.get(hmm_state, f'State {hmm_state}')} Proportion Across Genotypes, per zone"
    fig.suptitle(suptitle + " (hatched = not in zone set)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"all_genotypes_{filename_stem}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "groups_x_zones": one row per genotype group x zone (pooled over its sessions)
        source_data.record(pd.DataFrame(recorded).assign(metric=metric))
        source_data.flush("groups_x_zones", figure=save_path.stem)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close()  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig
