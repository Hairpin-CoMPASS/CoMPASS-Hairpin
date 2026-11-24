"""
BOUT-TYPE BASED STATE COMPARISONS
Author: Shreya Bangera
Goal:
    â”œâ”€â”€ Classifies bouts as successful or unsuccessful based on target reach.
    â”œâ”€â”€ Computes and compares HMM state proportions across these bout types.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING
from compass_labyrinth.utils import group_sample_sizes, append_n_to_legend, genotype_order

import compass_labyrinth.source_data as source_data


warnings.filterwarnings("ignore")


##################################################################
# Plot 4: Surveillance Probability by Bout Type
###################################################################
def assign_bout_indices(
    df: pd.DataFrame,
    delimiter_node: int = 0,
) -> pd.DataFrame:
    """
    Assign bout indices to each row in the dataframe based on delimiter nodes.
    Bout = delimiter_node --> Other non-entry nodes --> delimiter_node

    Parameters
    -----------
    df : pd.DataFrame
        Dataframe with 'session' and 'grid_number' columns.
    delimiter_node : int
        Grid Number that indicates the start of a new bout.
    """
    # Vectorized: within each session (original row order) bout_id is 1 + the
    # number of delimiter rows up to and including this one. Rows come back
    # grouped by session in sorted order, as the old per-row loop returned them.
    df = df.sort_values("session", kind="stable").reset_index(drop=True)
    is_delim = (df["grid_number"] == delimiter_node).astype("int64")
    df["bout_id"] = 1 + is_delim.groupby(df["session"]).cumsum()
    return df


def compute_surveillance_probabilities(
    df_hmm: pd.DataFrame,
    decision_nodes: str = "decision_reward",
    node_type_mapping: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute surveillance probability at Decision nodes by Bout type.
    - Successful -> reached the target at least once
    - Unsuccessful -> doesn't reached the target

    Parameters
    -----------
    df_hmm : pd.DataFrame
        Dataframe with 'genotype', 'session', 'hmm_state', 'grid_number', 'region', and 'bout_id'.
    decision_nodes : str
        Type of decision node to consider for surveillance probability -- a
        key in node_type_mapping, or constants.NODE_TYPE_MAPPING if not given.
    node_type_mapping : dict | None
        Custom {name: [grid_number, ...]} mapping to look decision_nodes up
        in, instead of the global constants.NODE_TYPE_MAPPING -- pass a
        notebook's own ZONES dict here to use custom zone groupings.

    Returns
    --------
    pd.DataFrame
        Dataframe with median surveillance probabilities per genotype, session, and bout type.
    """
    records = []
    mapping = node_type_mapping if node_type_mapping is not None else NODE_TYPE_MAPPING
    decision_nodes_ids = mapping.get(decision_nodes, [])

    for session_id, sess_df in df_hmm.groupby("session"):
        genotype = sess_df["genotype"].unique()[0]
        bouts = list(sess_df.groupby("bout_id"))[1:]  # skip incomplete first bout

        for bout_num, (_, bout_df) in enumerate(bouts, 1):
            success = "Successful" if "target_zone" in bout_df["region"].values else "Unsuccessful"
            state_probs = bout_df[bout_df["grid_number"].isin(decision_nodes_ids)]["hmm_state"].value_counts(
                normalize=True
            )
            prob_state_1 = state_probs.get(1, np.nan)

            records.append(
                {
                    "session": session_id,
                    "genotype": genotype,
                    "bout_no": bout_num,
                    "successful_bout": success,
                    "probability_1": prob_state_1,
                }
            )

    index_df = pd.DataFrame(records)
    median_df = (
        index_df.dropna().groupby(["genotype", "session", "successful_bout"])["probability_1"].median().reset_index()
    )
    return (index_df, median_df)


def plot_surveillance_by_bout(
    config: dict,
    median_df: pd.DataFrame,
    ylim: float,
    figure_size: tuple = (6, 6),
    palette: str | list = "Set2",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Barplot to depict the surveillance probabilities with t-test independent p-values.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    median_df : pd.DataFrame
        Dataframe with median surveillance probabilities per genotype, session, and bout type.
    ylim : float
        Y-axis limit for the plot.
    figure_size : tuple
        Size of the figure.
    palette : str | list
        Color palette for the plot. A single-color list can't distinguish
        multiple genotypes on the hue axis -- use a named qualitative palette
        (default) or an explicit list with one color per genotype.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    plt.Figure
        The generated matplotlib figure.
    """
    plt.figure(figsize=figure_size)
    genotypes = sorted(median_df["genotype"].unique())
    ax = sns.barplot(
        x="successful_bout",
        y="probability_1",
        hue="genotype",
        hue_order=genotype_order(median_df["genotype"]),
        data=median_df,
        errorbar="se",
        capsize=0.1,
        err_kws={"linewidth": 1.6},
        palette=palette,
        edgecolor="black",
    )

    plt.xlabel("bout_type", fontsize=12)
    plt.ylabel("Surveillance Probability (State 1)", fontsize=12)
    plt.title("Surveillance Probability across Bout Types", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, ylim)
    legend = plt.legend(title="Genotype", frameon=True, loc="upper right")
    append_n_to_legend(legend, group_sample_sizes(median_df, "genotype"))
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = (
            Path(config["project_path_full"]) / figures_dirname
            / f"surveillance_probability_by_bout{filename_suffix}.pdf"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_bout_outcome" -- one row per session per bout outcome
        # (Successful/Unsuccessful), named for its shape rather than this
        # step's own folder.
        source_data.record(median_df)
        source_data.flush("sessions_x_bout_outcome", figure=save_path.stem)
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


def test_within_genotype_success(index_df):
    """
    T-tests between Successful vs Unsuccessful bouts for each genotype.
    """
    results = []

    for genotype in sorted(index_df["genotype"].unique()):
        df = index_df[index_df["genotype"] == genotype]
        s = df[df["successful_bout"] == "Successful"]["probability_1"].dropna()
        u = df[df["successful_bout"] == "Unsuccessful"]["probability_1"].dropna()

        if len(s) >= 2 and len(u) >= 2:
            stat, pval = ttest_ind(s, u, equal_var=False)
            results.append(
                {
                    "genotype": genotype,
                    "group_1": "Successful",
                    "group_2": "Unsuccessful",
                    "t_stat": stat,
                    "p_value": pval,
                }
            )

    return pd.DataFrame(results)


def test_across_genotypes_per_bout(
    index_df: pd.DataFrame,
    bout_type: str = "Successful",
    value: str = "probability_1",
):
    """
    T-tests between genotypes for either Successful or Unsuccessful bouts
    (on the `value` column).
    """
    results = []

    df = index_df[index_df["successful_bout"] == bout_type]
    genotypes = sorted(df["genotype"].unique())

    for g1, g2 in combinations(genotypes, 2):
        vals1 = df[df["genotype"] == g1][value].dropna()
        vals2 = df[df["genotype"] == g2][value].dropna()

        if len(vals1) >= 2 and len(vals2) >= 2:
            stat, pval = ttest_ind(vals1, vals2, equal_var=False)
            results.append(
                {"bout_type": bout_type, "genotype_1": g1, "genotype_2": g2, "t_stat": stat, "p_value": pval}
            )

    return pd.DataFrame(results)


def run_within_genotype_mixedlm_with_fdr(median_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a mixed-effects model per genotype comparing Successful vs Unsuccessful bouts,
    with Session as a random effect. Applies FDR correction.

    Returns
    - DataFrame with genotype, Effect size, raw P-value, FDR P-value, and significance flag.
    """
    results = []
    genotypes = median_df["genotype"].unique()

    for genotype in genotypes:
        df_sub = median_df[median_df["genotype"] == genotype].copy()

        # Ensure sufficient sessions
        if df_sub["session"].nunique() < 2:
            continue

        # Ensure both bout types are present
        bout_counts = df_sub["successful_bout"].value_counts()
        if not all(x in bout_counts.index for x in ["Successful", "Unsuccessful"]):
            continue

        try:
            # Proper data typing
            df_sub["successful_bout"] = pd.Categorical(
                df_sub["successful_bout"], categories=["Unsuccessful", "Successful"]
            )
            df_sub["session"] = df_sub["session"].astype(str)

            # Fit model
            model = mixedlm(
                "probability_1 ~ successful_bout",
                data=df_sub,
                groups=df_sub["session"],
            )
            result = model.fit()

            # Extract stats
            term_name = next((k for k in result.params.keys() if "successful_bout" in k), None)
            coef = result.params.get(term_name, np.nan)
            pval = result.pvalues.get(term_name, np.nan)

        except Exception as e:
            coef = np.nan
            pval = np.nan

        results.append({"genotype": genotype, "effect_successful_vs_unsuccessful": coef, "p_value": pval})

    result_df = pd.DataFrame(results)

    # Apply FDR correction -- only over rows with a valid p_value (a failed
    # model fit above left some NaN), so one of those doesn't poison every
    # other row's fdr_p_value via multipletests.
    result_df["fdr_p_value"] = np.nan
    result_df["significant_fdr_005"] = False
    if not result_df.empty:
        valid = result_df["p_value"].notna()
        if valid.sum() > 0:
            reject, pvals_corrected, _, _ = multipletests(result_df.loc[valid, "p_value"], method="fdr_bh")
            result_df.loc[valid, "fdr_p_value"] = pvals_corrected
            result_df.loc[valid, "significant_fdr_005"] = reject

    return result_df


##################################################################
# Plot 4 by time bin
##################################################################
BOUT_OUTCOMES = ["Successful", "Unsuccessful"]


def surveillance_by_bout_bins(
    df_hmm: pd.DataFrame,
    decision_nodes: str,
    binning: str = "pct_bins",
    node_type_mapping: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """compute_surveillance_probabilities() with each bout in the time bin
    (time_bins.BINNINGS) of its first frame. Returns (index_df with
    'time_bin' 1..5, median per genotype x session x bout type x time_bin).
    Bouts starting past the last frame bin are dropped."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS, add_time_bins

    bin_col, _ = BINNINGS[binning]
    binned = add_time_bins(df_hmm)
    index_df, _ = compute_surveillance_probabilities(binned, decision_nodes, node_type_mapping)
    # same numbering as compute_surveillance_probabilities(): bouts in bout_id order, first one skipped
    start = binned.sort_values("s_no").groupby(["session", "bout_id"])[bin_col].first().reset_index()
    start = start.sort_values(["session", "bout_id"])
    start["bout_no"] = start.groupby("session").cumcount()
    index_df = index_df.merge(start.rename(columns={bin_col: "time_bin"})[["session", "bout_no", "time_bin"]],
                              on=["session", "bout_no"], how="left").dropna(subset=["time_bin"])
    index_df["time_bin"] = index_df["time_bin"].astype(int)
    median_df = (index_df.dropna(subset=["probability_1"])
                 .groupby(["genotype", "session", "successful_bout", "time_bin"])["probability_1"]
                 .median().reset_index())
    return index_df, median_df


def plot_surveillance_by_bout_bins(
    config: dict,
    median_df: pd.DataFrame,
    binning: str,
    ylim: tuple | None = None,
    palette: str | list = "Set2",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Successful | Unsuccessful panels (shared y): surveillance probability
    by time bin, one line per genotype group (mean +- SE across mice, each
    mouse's median over its bouts in that bin); a point with fewer mice than
    its group's n is labeled with its count. ylim (lo, hi); None = fit the data."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS, BINNING_TITLES

    _, titles = BINNINGS[binning]
    order = genotype_order(median_df["genotype"])
    colors = dict(zip(order, sns.color_palette(palette, len(order)) if isinstance(palette, str) else palette))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    offsets = np.linspace(-0.12, 0.12, len(order)) if len(order) > 1 else [0]
    group_n = group_sample_sizes(median_df, "genotype")
    lows, highs = [], []
    for ax, bout_type in zip(axes, BOUT_OUTCOMES):
        d = median_df[median_df["successful_bout"] == bout_type]
        for off, g in zip(offsets, order):
            st = d[d["genotype"] == g].groupby("time_bin")["probability_1"].agg(["mean", "sem", "count"])
            st = st.reindex(range(1, len(titles) + 1))
            x = np.arange(1, len(titles) + 1) + off
            ax.errorbar(x, st["mean"], yerr=st["sem"], marker="o", ms=5, capsize=3, lw=1.6,
                        color=colors[g], label=g)
            lows.append((st["mean"] - st["sem"].fillna(0)).min())
            highs.append((st["mean"] + st["sem"].fillna(0)).max())
            for xi, (m, c) in zip(x, zip(st["mean"], st["count"])):
                if pd.notna(m) and c < group_n.get(g, 0):
                    ax.annotate(f"n={int(c)}", (xi, m), textcoords="offset points", xytext=(5, -9),
                                ha="left", fontsize=7, color=colors[g])
        ax.set_xticks(range(1, len(titles) + 1), titles, fontsize=9)
        ax.set_title(f"{bout_type} bouts", fontsize=13)
        ax.set_xlabel(f"Time in session ({BINNING_TITLES[binning]})", fontsize=11)
        ax.grid(axis="y", color="0.92")
    if ylim is None:
        lo, hi = np.nanmin(lows), np.nanmax(highs)
        ylim = (max(0.0, np.floor(lo * 10 - 0.5) / 10), min(1.0, np.ceil(hi * 10 + 0.5) / 10))
    axes[0].set_ylim(*ylim)
    axes[0].set_ylabel("Surveillance Probability (State 1)", fontsize=12)
    fig.suptitle("Surveillance Probability by Bout Type over the session", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, title="Genotype", loc="lower center", ncol=len(order), frameon=False,
                        fontsize=9)
    append_n_to_legend(legend, group_n)

    if save_fig:
        save_path = (Path(config["project_path_full"]) / figures_dirname
                     / f"surveillance_probability_by_bout{filename_suffix}_{binning}.pdf")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_bout_outcome_x_time_bins": one row per session x bout outcome x time bin
        source_data.record(median_df.assign(time_bin=median_df["time_bin"].map(lambda b: titles[b - 1])))
        source_data.flush("sessions_x_bout_outcome_x_time_bins", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    if return_fig:
        return fig


def _bout_types(median_df: pd.DataFrame) -> list:
    """The 'successful_bout' values present: Successful, Unsuccessful, then any others (e.g. Plot 5c's to-reward)."""
    present = list(dict.fromkeys(median_df["successful_bout"]))
    return [b for b in BOUT_OUTCOMES if b in present] + [b for b in present if b not in BOUT_OUTCOMES]


def bout_bins_mixedlm(median_df: pd.DataFrame, value: str = "probability_1") -> pd.DataFrame:
    """Per bout type: mixed model value ~ genotype * time_bin
    (categorical) with session as random effect, fit by ML; likelihood-ratio
    tests for the genotype x time bin interaction (do groups change
    differently over the session?) and for genotype overall."""
    from scipy.stats import chi2

    def fit(formula, d):
        return mixedlm(formula, data=d, groups=d["session"]).fit(reml=False)

    rows = []
    for bout_type in _bout_types(median_df):
        d = median_df[median_df["successful_bout"] == bout_type].dropna(subset=[value]).copy()
        if d["genotype"].nunique() < 2:
            continue
        d["session"] = d["session"].astype(str)
        d["time_bin"] = d["time_bin"].astype(str)
        try:
            full = fit(f"{value} ~ C(genotype) * C(time_bin)", d)
            additive = fit(f"{value} ~ C(genotype) + C(time_bin)", d)
            no_geno = fit(f"{value} ~ C(time_bin)", d)
        except Exception:
            continue
        for term, big, small in [("genotype x time_bin", full, additive), ("genotype", additive, no_geno)]:
            lr = 2 * (big.llf - small.llf)
            df_ = len(big.fe_params) - len(small.fe_params)
            rows.append({"bout_type": bout_type, "term": term, "lr_stat": lr, "df": df_,
                         "p_value": chi2.sf(lr, df_), "n_mice": d["session"].nunique(), "n_points": len(d)})
    return pd.DataFrame(rows)


def bout_bins_ttests(median_df: pd.DataFrame, binning: str, value: str = "probability_1") -> pd.DataFrame:
    """test_across_genotypes_per_bout() in each time bin, for each bout type;
    FDR (Benjamini-Hochberg) across the bins within each bout type x genotype pair."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS

    _, titles = BINNINGS[binning]
    parts = []
    for bout_type in _bout_types(median_df):
        for b in sorted(median_df["time_bin"].unique()):
            t = test_across_genotypes_per_bout(median_df[median_df["time_bin"] == b], bout_type=bout_type, value=value)
            if len(t):
                parts.append(t.assign(time_bin=titles[int(b) - 1]))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["fdr_p_value"] = np.nan
    for _, idx in out.groupby(["bout_type", "genotype_1", "genotype_2"]).groups.items():
        out.loc[idx, "fdr_p_value"] = multipletests(out.loc[idx, "p_value"], method="fdr_bh")[1]
    return out[["bout_type", "time_bin", "genotype_1", "genotype_2", "t_stat", "p_value", "fdr_p_value"]]


##################################################################
# Plot 5c / 5d: Bout time and bout distance
##################################################################
# Parts of a successful bout, which add up to the whole bout -- Plot 5's 7 phases with the
# reward zone split into the first visit and later visits: panel name -> (part, x-axis label,
# color in the composition panel). bout_lengths() gives each part as 'frames_<part>' (time)
# and 'cells_<part>' (distance). Medians and tests use only bouts that have the part (a log
# axis can't show 0) -- matters for backtrack, re-entry, the overshoot.
SEGMENTS = {
    "Successful: outbound": ("outbound", "Outbound", "#2b8cbe"),
    "Successful: backtrack": ("backtrack", "Backtrack", "#a6bddb"),
    "Successful: reward, first visit": ("reward_first", "Reward:\nfirst visit", "#fec44f"),
    "Successful: overshoot forward": ("overshoot_forward", "Overshoot\nforward", "#ef3b2c"),
    "Successful: overshoot backward": ("overshoot_backward", "Overshoot\nbackward", "#fc9272"),
    "Successful: reward, later visits": ("reward_later", "Reward:\nlater visits", "#fee391"),
    "Successful: return": ("return", "Return", "#4d4d4d"),
    "Successful: re-entry": ("reentry", "Re-entry", "#bdbdbd"),
}
PHASE_PARTS = ["outbound", "backtrack", "overshoot_forward", "overshoot_backward", "return", "reentry"]
# Time-bin figure rows after Successful | Unsuccessful (SEGMENTS keys without "Successful: ")
BIN_PAIRS = ["outbound", "return", "reward, first visit", "reward, later visits", "backtrack", "re-entry",
             "overshoot forward", "overshoot backward"]
# What a bout is measured in: time (frames) or distance (cells moved along the path)
BOUT_MEASURES = {
    "frames": {"name": "Bout time", "unit": "frames", "share": "Where the time goes", "stem": "bout_length",
               "sheet": "sessions_x_bout_length"},
    "cells": {"name": "Bout distance", "unit": "cells moved", "share": "Where the distance goes",
              "stem": "bout_distance", "sheet": "sessions_x_bout_distance"},
}


def bout_lengths(
    df_hmm_bouts: pd.DataFrame,
    labeled: pd.DataFrame,
    target_region: str = "target_zone",
    overshoot_region: str = "overshoot",
    max_step: int = 3,
) -> pd.DataFrame:
    """One row per complete bout (entry zone -> maze -> entry zone; a bout cut
    off by its session's start or end is dropped): 'frames' (analyzed frames
    in the bout), 'cells' (path positions moved, frame to frame; jumps of more
    than max_step positions are tracking errors and count 0, as in Plot 5),
    'successful_bout' (reached target_region or not), the 'pct_bin' /
    'frame_bin' of its first frame (time_bins.add_time_bins()) and, for
    successful bouts (NaN otherwise), the SEGMENTS parts as 'frames_<part>'
    and 'cells_<part>', which add up to 'frames' / 'cells': Plot 5's
    outbound, backtrack, overshoot forward / backward, return, re-entry, and
    the reward zone ('at reward') split into 'reward_first' (its first stay,
    from first arrival until it first leaves) and 'reward_later' (the rest).
    A frame's movement counts toward the part of the frame it arrives in.
    Also by zone (frames): 'frames_to_reward' (before the first arrival),
    'frames_in_overshoot', 'frames_reward_to_start' (back on the path after
    the first arrival). `df_hmm_bouts` from assign_bout_indices_from_entry_zone()
    (bout_id 0 = entry-zone frames); `labeled` (frames with 'session', 's_no',
    'phase') from label_bout_directions()."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import add_time_bins
    from compass_labyrinth.post_hoc_analysis.level_1.direction_analysis import PATH_POSITION

    d = add_time_bins(df_hmm_bouts).merge(labeled[["session", "s_no", "phase"]], on=["session", "s_no"], how="left")
    d = d.sort_values(["session", "s_no"])
    by_bout = ["session", "bout_id"]
    d["pos"] = d.groupby(by_bout).cumcount() + 1
    path = d["grid_number"].astype(int).map(PATH_POSITION)
    step = path.groupby([d["session"], d["bout_id"]]).diff().abs().fillna(0)
    d["step_cells"] = step.where(step <= max_step, 0)
    d["in_target"] = d["region"] == target_region
    d["in_overshoot"] = d["region"] == overshoot_region
    in_maze = d[d["bout_id"] > 0]
    target = in_maze[in_maze["in_target"]].groupby(by_bout)["pos"]
    bouts = in_maze.groupby(by_bout).agg(
        genotype=("genotype", "first"), start=("s_no", "first"), end=("s_no", "last"), frames=("s_no", "size"),
        cells=("step_cells", "sum"), pct_bin=("pct_bin", "first"), frame_bin=("frame_bin", "first"))
    first = target.min()
    bouts["frames_to_reward"] = first - 1
    # Everything from the first arrival on, split by zone (all visits/trips together)
    m = in_maze.join(first.rename("first_target"), on=by_bout)
    after = m[m["pos"] >= m["first_target"]]
    first_leave = after[~after["in_target"]].groupby(by_bout)["pos"].min()
    bouts["frames_in_overshoot"] = after[after["in_overshoot"]].groupby(by_bout).size()
    bouts["frames_reward_to_start"] = after[~after["in_target"] & ~after["in_overshoot"]].groupby(by_bout).size()
    # Each frame's part: its phase, with the reward zone split at the first time it leaves
    m = m.join(first_leave.rename("first_leave"), on=by_bout)
    part = m["phase"].where(m["phase"].isin(PHASE_PARTS))
    part = part.mask(m["phase"] == "at_reward", np.where(m["pos"] < m["first_leave"], "reward_first", "reward_later"))
    per_part = m.assign(part=part).groupby(by_bout + ["part"]).agg(frames=("s_no", "size"), cells=("step_cells", "sum"))
    parts = [p for p, *_ in SEGMENTS.values()]
    for measure in ("frames", "cells"):
        bouts = bouts.join(per_part[measure].unstack("part").reindex(columns=parts).add_prefix(f"{measure}_"))
    bouts = bouts.reset_index()
    entry = d[d["bout_id"] == 0].groupby("session")["s_no"].agg(first_entry="min", last_entry="max")
    bouts = bouts.join(entry, on="session")
    bouts = bouts[(bouts["start"] > bouts["first_entry"]) & (bouts["end"] < bouts["last_entry"])]
    ok = bouts["frames_to_reward"].notna()
    bouts["successful_bout"] = np.where(ok, "Successful", "Unsuccessful")
    cols = [f"{ms}_{p}" for ms in ("frames", "cells") for p in parts] + ["frames_in_overshoot", "frames_reward_to_start"]
    bouts[cols] = bouts[cols].astype(float).fillna(0).where(ok)
    return bouts.drop(columns=["first_entry", "last_entry"]).reset_index(drop=True)


def bout_parts(bouts: pd.DataFrame, measure: str = "frames") -> pd.DataFrame:
    """bout_lengths() in long form, one row per bout x panel: Successful /
    Unsuccessful (whole bout) and each SEGMENTS part of the successful bouts,
    as 'successful_bout' (the panel) and 'value' (in `measure`, "frames" or
    "cells"; > 0 only)."""
    parts = [bouts.assign(value=bouts[measure])]
    s = bouts[bouts["successful_bout"] == "Successful"]
    for panel, (part, *_) in SEGMENTS.items():
        parts.append(s.assign(successful_bout=panel, value=s[f"{measure}_{part}"]))
    long = pd.concat(parts, ignore_index=True)
    return long[long["value"] > 0]


def bout_length_medians(bouts: pd.DataFrame, binning: str | None = None, measure: str = "frames") -> pd.DataFrame:
    """Per session x panel (bout_parts()) [x 'time_bin' of the bout's first
    frame, for `binning`]: the median over its bouts (column named `measure`),
    its log10 ('log10_<measure>', used by the tests) and the bout count ('bouts')."""
    by = []
    if binning is not None:
        from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS

        bouts = bouts.rename(columns={BINNINGS[binning][0]: "time_bin"}).dropna(subset=["time_bin"])
        bouts = bouts.assign(time_bin=bouts["time_bin"].astype(int))
        by = ["time_bin"]
    out = bout_parts(bouts, measure).groupby(["genotype", "session", "successful_bout", *by])["value"].agg(
        median="median", bouts="size").reset_index().rename(columns={"median": measure})
    out[f"log10_{measure}"] = np.log10(out[measure])
    return out


def _seconds_axis(ax, fps: float, which: str = "y") -> None:
    """A second axis in seconds (frames / fps) on the right (y) or top (x)."""
    conv = (lambda f: f / fps, lambda s: s * fps)
    if which == "y":
        ax.secondary_yaxis("right", functions=conv).set_ylabel(f"seconds (at {fps:g} fps)", fontsize=10)
    else:
        ax.secondary_xaxis("top", functions=conv).set_xlabel(f"seconds (at {fps:g} fps)", fontsize=10)


def plot_bout_length(
    config: dict,
    bouts: pd.DataFrame,
    median_df: pd.DataFrame,
    measure: str = "frames",
    fps: float = 5.0,
    palette: str | list = "Set2",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Bout time (measure="frames") or bout distance ("cells"), four panels:
    (1) the whole bout, Successful vs Unsuccessful, and (2) the parts of a
    successful bout (SEGMENTS) -- per mouse, the median over its bouts
    (dots), bars = mean +- SE across mice, log axis shared by both; (3) each
    part's share of the successful bouts' total (per mouse, then the mean
    across mice), stacked to 100%; (4) the distribution (cumulative) of
    successful bouts' totals per group, each mouse weighted equally. `bouts`
    from bout_lengths(), `median_df` from bout_length_medians(bouts, measure=measure)."""
    from matplotlib.gridspec import GridSpec

    info = BOUT_MEASURES[measure]
    order = genotype_order(median_df["genotype"])
    colors = dict(zip(order, sns.color_palette(palette, len(order)) if isinstance(palette, str) else palette))
    n = group_sample_sizes(median_df, "genotype")
    fig = plt.figure(figsize=(19, 13))
    gs = GridSpec(2, 10, figure=fig, hspace=0.45, wspace=1.2)
    ax_a, ax_b = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:10])
    ax_s, ax_c = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 5:10])  # legend between
    if measure == "frames":  # log: one axis for whole bouts and parts; linear: parts get their own
        ax_b.sharey(ax_a)
    segments = [p for p in SEGMENTS if p in set(median_df["successful_bout"])]
    for ax, panels, title in [(ax_a, BOUT_OUTCOMES, "Whole bout"), (ax_b, segments, "Parts of successful bouts")]:
        d = median_df[median_df["successful_bout"].isin(panels)]
        kw = dict(data=d, x="successful_bout", y=measure, order=panels, hue="genotype", hue_order=order,
                  palette=colors, ax=ax, legend=False)
        sns.barplot(**kw, errorbar="se", capsize=0.1, err_kws={"linewidth": 1.4}, edgecolor="black", alpha=0.85)
        sns.stripplot(**kw, dodge=True, size=4, edgecolor="black", linewidth=0.5)
        ax.set_yscale("log" if measure == "frames" else "linear")  # time is very skewed; distance less so
        ax.set_xlabel("")
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", which="major", color="0.92")
    ax_b.set_xticks(range(len(segments)), [SEGMENTS[p][1] for p in segments], fontsize=10)
    ax_a.set_ylabel(f"{info['name']} ({info['unit']}; per-mouse median)", fontsize=11)
    ax_b.set_ylabel("" if measure == "frames" else f"{info['unit'].capitalize()} (per-mouse median)", fontsize=11)
    if measure == "frames":
        _seconds_axis(ax_b, fps)

    # Each part's share of the successful bouts' total, stacked to 100%
    s = bouts[bouts["successful_bout"] == "Successful"]
    cols = [f"{measure}_{part}" for part, *_ in SEGMENTS.values()]
    per_mouse = s.groupby(["session", "genotype"])[[*cols, measure]].sum()
    shares = 100 * per_mouse[cols].div(per_mouse[measure], axis=0)
    mean_share = shares.groupby("genotype").mean().reindex(order)
    bottom = np.zeros(len(order))
    for col, (_, label, color) in zip(cols, SEGMENTS.values()):
        v = mean_share[col].to_numpy()
        ax_s.bar(range(len(order)), v, bottom=bottom, color=color, edgecolor="black", linewidth=0.6,
                 label=label.replace("\n", " "))
        for xi, (b0, vi) in enumerate(zip(bottom, v)):
            if vi >= 5:
                ax_s.text(xi, b0 + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=8)
        bottom += v
    ax_s.set_xticks(range(len(order)), order, fontsize=9)
    ax_s.set_ylim(0, 100)
    ax_s.set_ylabel(f"% of successful-bout {info['unit']} (mean across mice)", fontsize=11)
    ax_s.set_title(info["share"], fontsize=13)
    ax_s.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    # Cumulative distribution of successful bouts' totals, each mouse weighted equally
    for g in order:
        sg = s[s["genotype"] == g].sort_values(measure)
        if sg.empty:
            continue
        w = 1 / sg.groupby("session")[measure].transform("size")
        ax_c.step(sg[measure], w.cumsum() / w.sum(), where="post", color=colors[g], lw=2, label=g)
    ax_c.set_xscale("log" if measure == "frames" else "linear")
    ax_c.set_ylim(0, 1.01)
    ax_c.set_xlabel(f"Successful bout: {info['name'].split()[1]} ({info['unit']})", fontsize=11)
    ax_c.set_ylabel("Fraction of bouts (each mouse weighted equally)", fontsize=11)
    ax_c.set_title("Successful bouts: distribution", fontsize=13)
    ax_c.grid(color="0.92")
    if measure == "frames":
        _seconds_axis(ax_c, fps, which="x")

    fig.suptitle(f"{info['name']} by outcome", fontsize=15, weight="bold", y=0.95)
    handles, labels = ax_c.get_legend_handles_labels()
    legend = fig.legend(handles, labels, title="Genotype", loc="lower center", ncol=len(order), frameon=False,
                        fontsize=9)
    append_n_to_legend(legend, n)

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{info['stem']}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # one row per session x panel (Successful / Unsuccessful / each part);
        # "_bouts": one row per successful bout (the distribution panel); "_shares": the stacked panel
        source_data.record(median_df)
        source_data.record(s[["session", "genotype", "bout_id", measure, *cols]], part="bouts")
        source_data.record(shares.reset_index(), part="shares")
        source_data.flush(info["sheet"], figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    if return_fig:
        return fig


def plot_bout_length_bins(
    config: dict,
    median_df: pd.DataFrame,
    binning: str,
    measure: str = "frames",
    fps: float = 5.0,
    palette: str | list = "Set2",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Bout time or distance by time bin of the bout's first frame, four
    panels per row (shared log y): Successful, Unsuccessful, then the parts
    in BIN_PAIRS order; one line per genotype group (mean +- SE across mice
    of each mouse's median in that bin); a point with fewer mice than its
    group's n is labeled with its count. `median_df` from
    bout_length_medians(bouts, binning, measure)."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS, BINNING_TITLES

    info = BOUT_MEASURES[measure]
    _, titles = BINNINGS[binning]
    order = genotype_order(median_df["genotype"])
    colors = dict(zip(order, sns.color_palette(palette, len(order)) if isinstance(palette, str) else palette))
    # Four panels per row: outcomes, then the parts in pairs
    panels = [*BOUT_OUTCOMES, *[f"Successful: {p}" for p in BIN_PAIRS]]
    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows), sharey=measure == "frames")
    axes = axes.ravel()
    offsets = np.linspace(-0.12, 0.12, len(order)) if len(order) > 1 else [0]
    group_n = group_sample_sizes(median_df, "genotype")
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        title = f"{panel} bouts" if panel in BOUT_OUTCOMES else "Successful: " + SEGMENTS[panel][1].replace("\n", " ")
        d = median_df[median_df["successful_bout"] == panel]
        for off, g in zip(offsets, order):
            st = d[d["genotype"] == g].groupby("time_bin")[measure].agg(["mean", "sem", "count"])
            st = st.reindex(range(1, len(titles) + 1))
            x = np.arange(1, len(titles) + 1) + off
            ax.errorbar(x, st["mean"], yerr=st["sem"], marker="o", ms=5, capsize=3, lw=1.6, color=colors[g], label=g)
            for xi, m, c in zip(x, st["mean"], st["count"]):
                if pd.notna(m) and c < group_n.get(g, 0):
                    ax.annotate(f"n={int(c)}", (xi, m), textcoords="offset points", xytext=(5, -9),
                                ha="left", fontsize=7, color=colors[g])
        ax.set_yscale("log" if measure == "frames" else "linear")
        ax.set_xticks(range(1, len(titles) + 1), titles, fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f"Time in session ({BINNING_TITLES[binning]})" if i + ncols >= len(panels) else "", fontsize=11)
        ax.grid(axis="y", which="major", color="0.92")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[0::ncols]:
        ax.set_ylabel(f"{info['unit'].capitalize()} (per-mouse median)", fontsize=11)
    if measure == "frames":
        for ax in [*axes[ncols - 1:len(panels):ncols], axes[len(panels) - 1]]:
            _seconds_axis(ax, fps)
    fig.suptitle(f"{info['name']} by outcome over the session", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), h_pad=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, title="Genotype", loc="lower center", ncol=len(order), frameon=False,
                        fontsize=9)
    append_n_to_legend(legend, group_n)

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{info['stem']}{filename_suffix}_{binning}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # one row per session x panel x time bin
        source_data.record(median_df.assign(time_bin=median_df["time_bin"].map(lambda b: titles[b - 1])))
        source_data.flush(f"{info['sheet']}_x_time_bins", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    if return_fig:
        return fig


def bout_length_ttests(median_df: pd.DataFrame, measure: str = "frames") -> pd.DataFrame:
    """Welch t-tests between every pair of genotype groups on log10 of the
    per-mouse median (time or distance), for each panel (Successful,
    Unsuccessful, each SEGMENTS part); FDR (Benjamini-Hochberg) across all
    rows of the table."""
    parts = [t for b in _bout_types(median_df)
             if len(t := test_across_genotypes_per_bout(median_df, bout_type=b, value=f"log10_{measure}"))]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["fdr_p_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def bout_length_mixedlm(bouts: pd.DataFrame, measure: str = "frames") -> pd.DataFrame:
    """Per bout (not per-mouse medians), mixed models on log10 time or
    distance with session as random effect, fit by ML, likelihood-ratio
    tests: whole bout ~ genotype * outcome -- the genotype x outcome
    interaction (does a group's successful-vs-unsuccessful gap differ?) and
    genotype; each SEGMENTS part ~ genotype (successful bouts with that
    part) -- genotype. Bouts with a 0 total are left out."""
    from scipy.stats import chi2

    def fit(formula, d):
        return mixedlm(formula, data=d, groups=d["session"]).fit(reml=False)

    if bouts["genotype"].nunique() < 2:
        return pd.DataFrame()
    d = bouts[bouts[measure] > 0]
    d = d.assign(log10_total=np.log10(d[measure]), session=d["session"].astype(str))
    parts = bout_parts(d, measure).assign(log10_value=lambda t: np.log10(t["value"]))
    tests = [  # (measure, term, data, full model, model without the term)
        ("whole bout", "genotype x outcome", d, "log10_total ~ C(genotype) * C(successful_bout)",
         "log10_total ~ C(genotype) + C(successful_bout)"),
        ("whole bout", "genotype", d, "log10_total ~ C(genotype) + C(successful_bout)",
         "log10_total ~ C(successful_bout)"),
    ] + [(panel, "genotype", parts[parts["successful_bout"] == panel], "log10_value ~ C(genotype)",
          "log10_value ~ 1") for panel in SEGMENTS]
    rows = []
    for name, term, data, big_f, small_f in tests:
        if data.empty or data["genotype"].nunique() < 2:
            continue
        try:
            big, small = fit(big_f, data), fit(small_f, data)
        except Exception:
            continue
        lr = 2 * (big.llf - small.llf)
        df_ = len(big.fe_params) - len(small.fe_params)
        rows.append({"measure": name, "term": term, "lr_stat": lr, "df": df_, "p_value": chi2.sf(lr, df_),
                     "n_mice": data["session"].nunique(), "n_bouts": len(data)})
    return pd.DataFrame(rows)
