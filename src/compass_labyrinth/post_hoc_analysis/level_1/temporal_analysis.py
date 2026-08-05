"""
TEMPORAL EVOLUTION OF STATE PROBABILITIES
Author: Shreya Bangera
Goal:
    ├── Tracks probability of being in a chosen HMM state over time bins per genotype and session-averaged plots

"""

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statistics import median
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING


warnings.filterwarnings("ignore")


def compute_node_state_medians_over_time(
    df_hmm: pd.DataFrame,
    state_types: list,
    lower_lim: int,
    upper_lim: int,
    bin_size: int,
    zone_names: list[str] = ["corner_reward", "nondecision_reward"],
    zone_labels: dict[str, str] | None = None,
    node_type_mapping: dict | None = None,
) -> pd.DataFrame:
    """
    Compute time-binned medians of HMM state proportions for a set of zones.

    Any number of zones can be compared (not just two) -- each key in
    zone_names (looked up in node_type_mapping, or constants.NODE_TYPE_MAPPING
    if not given) gets its own median-probability curve in the output.

    Parameters
    -----------
    df_hmm : pd.DataFrame
        Dataframe with 'genotype', 'session', 'hmm_state', and 'grid_number'.
    state_types : list
        List of HMM states to compute proportions for (e.g., [2])
    lower_lim : int
        Range of rows to consider per session
    upper_lim : int
        Range of rows to consider per session
    bin_size : int
        Number of rows per time bin
    zone_names : list[str]
        Keys (in node_type_mapping, or constants.NODE_TYPE_MAPPING if not given)
        for the zones to compare -- one median-probability curve per zone.
    zone_labels : dict[str, str] | None
        Optional {zone_name: display_label} overrides for the output's
        'node_type' column -- zones not present default to their own name.
    node_type_mapping : dict | None
        Custom {name: [grid_number, ...]} mapping to look zone_names up in,
        instead of the global constants.NODE_TYPE_MAPPING -- pass a notebook's
        own ZONES dict here to use custom zone groupings.

    Returns
    --------
    pd.DataFrame
        Dataframe with median proportions per session, time bin, node type, and genotype
    """
    mapping = node_type_mapping if node_type_mapping is not None else NODE_TYPE_MAPPING
    labels = zone_labels or {}
    li_node_genotype = []
    for genotype in df_hmm["genotype"].unique():
        sess_li = [x for _, x in df_hmm[df_hmm["genotype"] == genotype].groupby("session")]

        for zone in zone_names:
            node_type_list = mapping.get(zone, [])
            node_type_label = labels.get(zone, zone)
            med_df = pd.DataFrame(columns=["time_bins", "session", "median_probability"])

            row_index = 1
            for k in range(lower_lim, upper_lim, bin_size):
                for session_df in sess_li:
                    session_df = session_df.reset_index(drop=True)
                    df_subset = session_df.iloc[k : k + bin_size, :]

                    # Count by HMM state
                    st_cnt = df_subset.groupby(["grid_number", "hmm_state"]).size().rename("cnt").reset_index()
                    gn_cnt = df_subset.groupby(["grid_number"]).size().rename("tot").reset_index()
                    x_y = df_subset.groupby(["grid_number"]).agg({"x": "mean", "y": "mean"}).reset_index()

                    state_count = st_cnt.merge(gn_cnt, on="grid_number", how="left")
                    state_count["prop"] = state_count["cnt"] / state_count["tot"]
                    state_count = state_count.merge(x_y, on="grid_number", how="left")
                    subset = state_count[
                        (state_count["hmm_state"].isin(state_types)) & (state_count["grid_number"].isin(node_type_list))
                    ]

                    if not subset.empty:
                        med_val = median(subset["prop"])
                    else:
                        med_val = 0

                    med_df.loc[row_index, "median_probability"] = med_val
                    med_df.loc[row_index, "session"] = session_df["session"].unique()[0]
                    med_df.loc[row_index, "time_bins"] = k + bin_size
                    row_index += 1

            med_df = med_df[med_df["median_probability"] != 0]
            med_df["node_type"] = node_type_label
            med_df["genotype"] = genotype
            li_node_genotype.append(med_df)

    deci_df = pd.concat(li_node_genotype).reset_index(drop=True)
    deci_df["genotype_node_type"] = deci_df["genotype"] + " , " + deci_df["node_type"]
    deci_df["time_bins"] = deci_df["time_bins"].astype(int)

    return deci_df


def compute_node_velocity_medians_over_time(
    df_hmm: pd.DataFrame,
    lower_lim: int,
    upper_lim: int,
    bin_size: int,
    zone_names: list[str] = ["corner_reward", "nondecision_reward"],
    zone_labels: dict[str, str] | None = None,
    node_type_mapping: dict | None = None,
) -> pd.DataFrame:
    """
    Compute time-binned medians of per-grid-cell mean velocity for a set of
    zones -- the velocity counterpart of compute_node_state_medians_over_time():
    same per-session, per-time-bin, per-zone loop, but each bin's per-grid-cell
    value is a mean velocity instead of an HMM-state proportion, and the
    within-zone median is taken over those cell-mean velocities.

    Parameters
    -----------
    df_hmm : pd.DataFrame
        Dataframe with 'genotype', 'session', 'velocity', and 'grid_number'.
    lower_lim : int
        Range of rows to consider per session
    upper_lim : int
        Range of rows to consider per session
    bin_size : int
        Number of rows per time bin
    zone_names : list[str]
        Keys (in node_type_mapping, or constants.NODE_TYPE_MAPPING if not given)
        for the zones to compare -- one median-velocity curve per zone.
    zone_labels : dict[str, str] | None
        Optional {zone_name: display_label} overrides for the output's
        'node_type' column -- zones not present default to their own name.
    node_type_mapping : dict | None
        Custom {name: [grid_number, ...]} mapping to look zone_names up in,
        instead of the global constants.NODE_TYPE_MAPPING -- pass a notebook's
        own ZONES dict here to use custom zone groupings.

    Returns
    --------
    pd.DataFrame
        Dataframe with median velocity (column "median_velocity") per session,
        time bin, node type, and genotype -- pass to plot_node_state_median_curve()/
        plot_node_state_median_curves_by_zone() with value_col="median_velocity".
    """
    mapping = node_type_mapping if node_type_mapping is not None else NODE_TYPE_MAPPING
    labels = zone_labels or {}
    li_node_genotype = []
    for genotype in df_hmm["genotype"].unique():
        sess_li = [x for _, x in df_hmm[df_hmm["genotype"] == genotype].groupby("session")]

        for zone in zone_names:
            node_type_list = mapping.get(zone, [])
            node_type_label = labels.get(zone, zone)
            med_df = pd.DataFrame(columns=["time_bins", "session", "median_velocity"])

            row_index = 1
            for k in range(lower_lim, upper_lim, bin_size):
                for session_df in sess_li:
                    session_df = session_df.reset_index(drop=True)
                    df_subset = session_df.iloc[k : k + bin_size, :]

                    vel_by_grid = df_subset.groupby("grid_number")["velocity"].mean().reset_index()
                    subset = vel_by_grid[vel_by_grid["grid_number"].isin(node_type_list)]

                    if not subset.empty:
                        med_val = median(subset["velocity"])
                    else:
                        med_val = 0

                    med_df.loc[row_index, "median_velocity"] = med_val
                    med_df.loc[row_index, "session"] = session_df["session"].unique()[0]
                    med_df.loc[row_index, "time_bins"] = k + bin_size
                    row_index += 1

            med_df = med_df[med_df["median_velocity"] != 0]
            med_df["node_type"] = node_type_label
            med_df["genotype"] = genotype
            li_node_genotype.append(med_df)

    deci_df = pd.concat(li_node_genotype).reset_index(drop=True)
    deci_df["genotype_node_type"] = deci_df["genotype"] + " , " + deci_df["node_type"]
    deci_df["time_bins"] = deci_df["time_bins"].astype(int)

    return deci_df


def build_genotype_zone_palette(
    deci_df: pd.DataFrame,
    node_type_col: str = "node_type",
    genotype_col: str = "genotype",
    zone_palette: str = "tab10",
) -> dict:
    """
    Build a {genotype_node_type: color} palette where each zone gets its own
    base hue and genotypes within that zone are distinguished by shades
    (light -> dark) of that same hue, instead of an arbitrary flat color per
    genotype/zone combination.

    Parameters
    -----------
    deci_df : pd.DataFrame
        Output of compute_node_state_medians_over_time() -- must have
        node_type_col and genotype_col columns.
    node_type_col : str
        Column identifying each zone (one base hue per unique value).
    genotype_col : str
        Column identifying each genotype (one shade per unique value, within
        a zone's hue).
    zone_palette : str
        Named qualitative palette to draw zones' base hues from.

    Returns
    --------
    dict
        {f"{genotype} , {node_type}": color} -- keys match the
        'genotype_node_type' column produced by compute_node_state_medians_over_time().
    """
    zones = sorted(deci_df[node_type_col].unique())
    base_colors = sns.color_palette(zone_palette, n_colors=len(zones))
    palette = {}
    for zone, base_color in zip(zones, base_colors):
        genotypes = sorted(deci_df.loc[deci_df[node_type_col] == zone, genotype_col].unique())
        # light_palette runs light -> saturated; drop only the lightest shade
        # (too close to white to read) so the rest span as wide a range as
        # possible -- keeping the fully-saturated base_color as the darkest
        # genotype maximizes contrast between genotypes sharing this hue.
        shades = sns.light_palette(base_color, n_colors=len(genotypes) + 1)[1:]
        for genotype, shade in zip(genotypes, shades):
            palette[f"{genotype} , {zone}"] = shade
    return palette


def plot_node_state_median_curve(
    config: dict,
    deci_df: pd.DataFrame,
    figure_ylimit: tuple = (0.0, 1.1),
    palette: str | list | dict = "tab10",
    fig_title: str | None = None,
    col: str | None = None,
    value_col: str = "median_probability",
    metric_label: str | None = None,
    filename_stem: str | None = None,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot time-binned medians (or any other per-time-bin value) for a set of zones.

    Parameters
    -----------
    deci_df : pd.DataFrame
        DataFrame containing 'time_bins', value_col, and 'genotype_node_type'
        -- output of compute_node_state_medians_over_time() (or another
        compute_*() sharing its shape, e.g. compute_node_velocity_medians_over_time()).
    palette : str | list | dict
        Named qualitative palette (default), an explicit list of colors (one
        per unique 'genotype_node_type' category -- a too-short list makes
        multiple categories render in the same color), or a
        {genotype_node_type: color} dict, e.g. from build_genotype_zone_palette()
        to color by zone with genotype-distinguishing shades.
    figure_ylimit : tuple
        Tuple like (0, 0.6) for y-axis limits
    fig_title : str | None
        Title of the figure
    col : str | None
        Optional column to facet into side-by-side panels (e.g. a 'state_label'
        column so all HMM states can be shown on one figure instead of forcing
        them into the same 8-line genotype_node_type comparison).
    value_col : str
        Column on deci_df to plot on the y-axis -- "median_probability"
        (default, from compute_node_state_medians_over_time()) for HMM state
        proportions, or whatever column another compute_*() produced (e.g.
        "median_velocity").
    metric_label : str | None
        Overrides the state-based y-axis label entirely, e.g. "Median
        Velocity" -- use this for any value_col that isn't an HMM state
        proportion.
    filename_stem : str | None
        Overrides the state-based filename stem entirely, e.g.
        "temporal_median_velocity_curve" -- use alongside metric_label so the
        saved file is identifiable on its own, without relying on figures_dirname.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    plt.Figure
        Seaborn FacetGrid object
    """

    g = sns.catplot(
        data=deci_df,
        x="time_bins",
        y=value_col,
        hue="genotype_node_type",
        col=col,
        kind="point",
        errorbar="se",
        capsize=0.15,
        err_kws={"linewidth": 1.5},
        palette=palette,
        aspect=1.9,
        legend=True,  # auto legend
    )

    g.set(ylim=figure_ylimit)
    g.set_xlabels("Time/Frame Bins", size=12)
    g.set_ylabels(metric_label if metric_label else "Median Probability of HMM state", size=12)
    g.set_xticklabels(rotation=45, size=9, color="black")
    g.set_yticklabels(size=9, color="black")

    # Adjust legend appearance (not manual mapping!)
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 1))
        g._legend.set_frame_on(True)
        g._legend.set_title("")
    # suptitle (not plt.title) -- plt.title only labels a single subplot when
    # faceted with col=, leaving the rest of the figure untitled
    g.figure.suptitle(fig_title, fontsize=15)
    g.figure.tight_layout()

    # Save figure
    fig = g.figure
    if save_fig:
        stem = filename_stem if filename_stem else "temporal_median_state_probability_curve"
        save_path = (
            Path(config["project_path_full"]) / figures_dirname
            / f"{stem}{filename_suffix}.pdf"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


def plot_node_state_median_curves_by_zone(
    config: dict,
    deci_df: pd.DataFrame,
    zone_names: list[str] | None = None,
    figure_ylimit: tuple = (0.0, 1.1),
    palette: str | list | dict = "tab10",
    fig_title: str | None = None,
    value_col: str = "median_probability",
    metric_label: str | None = None,
    filename_stem: str | None = None,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
) -> None:
    """
    Like plot_node_state_median_curve(), but draws one figure per zone instead
    of overlaying every zone's genotype curves together -- illegible once
    zone_names has more than ~3 entries. Each zone's figure is hued by
    genotype alone (a single zone only ever has that zone's genotypes to
    distinguish), and all zone figures are written to one multi-page PDF.

    Parameters
    -----------
    deci_df : pd.DataFrame
        Output of compute_node_state_medians_over_time() (or another
        compute_*() sharing its shape, e.g. compute_node_velocity_medians_over_time())
        -- must have 'time_bins', value_col, 'node_type', and 'genotype'.
    zone_names : list[str] | None
        Order to draw zone pages in (defaults to deci_df['node_type']'s
        unique values in their existing order).
    palette : str | list | dict
        Named qualitative palette, an explicit list of colors, or a
        {genotype_node_type: color} dict (e.g. from build_genotype_zone_palette())
        -- a dict is sliced down to {genotype: color} per zone so colors stay
        consistent with the combined Plot 3 figure.
    value_col : str
        Column on deci_df to plot on the y-axis -- "median_probability"
        (default) for HMM state proportions, or whatever column another
        compute_*() produced (e.g. "median_velocity").
    metric_label : str | None
        Overrides the state-based y-axis label entirely, e.g. "Median
        Velocity" -- use this for any value_col that isn't an HMM state
        proportion.
    filename_stem : str | None
        Overrides the state-based filename stem entirely, e.g.
        "temporal_median_velocity_curve_by_zone" -- use alongside
        metric_label so the saved file is identifiable on its own, without
        relying on figures_dirname.
    Other parameters mirror plot_node_state_median_curve().
    """
    zones = zone_names if zone_names is not None else list(dict.fromkeys(deci_df["node_type"]))

    save_path = None
    pdf = None
    if save_fig:
        stem = filename_stem if filename_stem else "temporal_median_state_probability_curve_by_zone"
        save_path = (
            Path(config["project_path_full"]) / figures_dirname
            / f"{stem}{filename_suffix}.pdf"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(save_path)

    for zone in zones:
        zone_df = deci_df[deci_df["node_type"] == zone]
        if zone_df.empty:
            continue

        zone_palette = palette
        if isinstance(palette, dict):
            zone_palette = {
                key.split(" , ")[0]: color
                for key, color in palette.items()
                if key.split(" , ")[1] == zone
            }

        g = sns.catplot(
            data=zone_df,
            x="time_bins",
            y=value_col,
            hue="genotype",
            kind="point",
            errorbar="se",
            capsize=0.15,
            err_kws={"linewidth": 1.5},
            palette=zone_palette,
            aspect=1.9,
            legend=True,
        )

        g.set(ylim=figure_ylimit)
        g.set_xlabels("Time/Frame Bins", size=12)
        g.set_ylabels(metric_label if metric_label else "Median Probability of HMM state", size=12)
        g.set_xticklabels(rotation=45, size=9, color="black")
        g.set_yticklabels(size=9, color="black")

        if g._legend:
            g._legend.set_bbox_to_anchor((1.02, 1))
            g._legend.set_frame_on(True)
            g._legend.set_title("")

        g.figure.suptitle(f"{fig_title} -- {zone}" if fig_title else zone, fontsize=15)
        g.figure.tight_layout()

        if pdf is not None:
            pdf.savefig(g.figure, bbox_inches="tight", dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close(g.figure)

    if pdf is not None:
        pdf.close()
        print(f"Figure saved at: {save_path}")


def get_max_session_row_bracket(
    df_combined: pd.DataFrame,
    session_col: str = "session",
) -> int:
    """
    Finds the session with the maximum number of rows and returns the largest lower multiple of 10,000.

    Parameters
    -----------
    df_combined : pd.DataFrame
        Combined dataframe containing multiple sessions.
    session_col : str
        Name of the column representing session ID.

    Returns
    --------
    int
        Lower bracketed row count (e.g., 20000 if max session has 23567 rows).
    """
    session_counts = df_combined[session_col].value_counts()
    max_rows = session_counts.max()
    return int(np.floor(max_rows / 10000) * 10000)


def get_min_session_row_bracket(
    df_combined: pd.DataFrame,
    session_col: str = "session",
) -> int:
    """
    Finds the session with the minimum number of rows and returns the largest lower multiple of 10,000.

    Parameters
    -----------
    df_combined : pd.DataFrame
        Combined dataframe containing multiple sessions.
    session_col : str
        Name of the column representing session ID.

    Returns
    --------
    int
        Lower bracketed row count (e.g., 10000 if min session has 10234 rows).
    """
    session_counts = df_combined[session_col].value_counts()
    min_rows = session_counts.min()
    return int(np.floor(min_rows / 10000) * 10000)
