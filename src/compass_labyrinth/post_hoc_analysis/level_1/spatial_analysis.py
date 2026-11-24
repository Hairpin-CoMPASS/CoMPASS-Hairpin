"""
STATE DISTRIBUTIONS BY NODE-TYPE AND REGION
Author: Shreya Bangera
Goal:
    ├── Comparison of proportion of time spent in a state across Maze regions and Node types.
    ├── Allows genotype level comparisons behavioral states.
"""

from pathlib import Path
from itertools import combinations
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING
from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import HMM_STATE_LABELS
from compass_labyrinth.utils import group_sample_sizes, append_n_to_legend, genotype_order

import compass_labyrinth.source_data as source_data


warnings.simplefilter(action="ignore", category=FutureWarning)


##################################################################
# Plot 2: Probability of Surveillance across Node Types and Regions
###################################################################
def compute_state_probability(
    df_hmm: pd.DataFrame,
    column_of_interest: str,
    values_displayed: list[str] | None = None,
    state: int = 1,
) -> pd.DataFrame:
    """
    Computes HMM state proportions by category (e.g., node_type or region).
    Optionally reassigns decision node labels for 3-way and 4-way decisions.

    Parameters
    -----------
    df_hmm: pd.DataFrame
        Dataframe with 'genotype', 'session', 'hmm_state', and category column.
    column_of_interest: str
        'node_type' or 'region'.
    values_displayed: Optional[List[str]]
        Categories to include and order
    state: int
        HMM_state of interest

    Returns
    --------
    pd.DataFrame
        Dataframe with proportions per session.
    """

    df_plot = df_hmm.copy()

    # Optional reassignment of node_type for 3-way / 4-way decisions
    decision_3way_grids = NODE_TYPE_MAPPING.get("decision_3way", [])
    decision_4way_grids = NODE_TYPE_MAPPING.get("decision_4way", [])
    if column_of_interest == "node_type" and decision_3way_grids and decision_4way_grids:
        df_plot.loc[df_plot["grid_number"].isin(decision_3way_grids), "node_type"] = "3-way Decision (Reward)"
        df_plot.loc[df_plot["grid_number"].isin(decision_4way_grids), "node_type"] = "4-way Decision (Reward)"
        df_plot = df_plot.loc[~df_plot["node_type"].isin(["Entry Nodes", "Target Nodes"])]

    # Compute state occurrence counts
    st_cnt = (
        df_plot.groupby(["genotype", column_of_interest, "session", "hmm_state"]).size().rename("cnt").reset_index()
    )
    gn_cnt = df_plot.groupby(["genotype", column_of_interest, "session"]).size().rename("tot").reset_index()
    state_count = st_cnt.merge(gn_cnt, on=[column_of_interest, "genotype", "session"], how="left")
    state_count["prop"] = state_count["cnt"] / state_count["tot"]

    # Filter for target HMM state and reorder
    state_count = state_count[state_count["hmm_state"] == state].copy()
    if values_displayed:
        state_count = state_count[state_count[column_of_interest].isin(values_displayed)].reset_index(drop=True)
        state_count[column_of_interest] = pd.Categorical(
            state_count[column_of_interest], categories=values_displayed, ordered=True
        )

    return state_count


def compute_zone_velocity(
    df_hmm: pd.DataFrame,
    column_of_interest: str,
    values_displayed: list[str] | None = None,
) -> pd.DataFrame:
    """
    Computes mean velocity by category (e.g., custom_zone), per session -- the
    velocity counterpart of compute_state_probability(), for Plot 2b's "Mean
    Velocity across Custom Zones" boxplot instead of a state-proportion one.

    Parameters
    -----------
    df_hmm: pd.DataFrame
        Dataframe with 'genotype', 'session', 'velocity', and category column.
    column_of_interest: str
        e.g. 'custom_zone'.
    values_displayed: Optional[List[str]]
        Categories to include and order.

    Returns
    --------
    pd.DataFrame
        Dataframe with mean velocity per session, genotype, and category
        (column "velocity") -- pass to plot_state_probability_boxplot() with
        value_col="velocity" and to run_pairwise_ttests() with
        value_col="velocity".
    """
    velocity_by_zone = (
        df_hmm.groupby(["genotype", column_of_interest, "session"])["velocity"].mean().reset_index()
    )
    if values_displayed:
        velocity_by_zone = velocity_by_zone[velocity_by_zone[column_of_interest].isin(values_displayed)].reset_index(
            drop=True
        )
        velocity_by_zone[column_of_interest] = pd.Categorical(
            velocity_by_zone[column_of_interest], categories=values_displayed, ordered=True
        )
    return velocity_by_zone


def plot_state_probability_boxplot(
    config: dict,
    state_count_df: pd.DataFrame,
    column_of_interest: str,
    state: int = 1,
    value_col: str = "prop",
    metric_label: str | None = None,
    filename_stem: str | None = None,
    figsize: tuple = (16, 7),
    palette: str = "Set2",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plots boxplot of HMM state probabilities (or any other per-category value) by category and genotype.

    Parameters
    state_count: pd.DataFrame
        Dataframe returned from compute_state_probability() (or another
        compute_*() sharing its ["genotype", column_of_interest, value_col]
        shape, e.g. compute_zone_velocity()).
    column_of_interest: str
        Categorical variable on x-axis
    state: int
        HMM state used for labeling. Ignored (label falls back to
        metric_label) when metric_label is given.
    value_col : str
        Column on state_count_df to plot on the y-axis -- "prop" (default,
        from compute_state_probability()) for HMM state proportions, or
        whatever column another compute_*() produced (e.g. "velocity").
    metric_label : str | None
        Overrides the state-based y-axis label entirely, e.g. "Mean
        Velocity" -- use this for any value_col that isn't an HMM state
        proportion.
    filename_stem : str | None
        Overrides the state-based filename stem entirely, e.g.
        "mean_velocity" -- use alongside metric_label so the saved file is
        identifiable on its own, without relying on figures_dirname.
    figsize: tuple
        Figure size
    palette: str
        Seaborn palette
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
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(
        x=column_of_interest,
        y=value_col,
        hue="genotype",
        hue_order=genotype_order(state_count_df["genotype"]),
        data=state_count_df,
        palette=palette,
    )
    append_n_to_legend(ax.get_legend(), group_sample_sizes(state_count_df, "genotype"))
    state_name = HMM_STATE_LABELS.get(state, f"State {state}")
    ylabel = metric_label if metric_label else f"Probability of {state_name} state"
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(column_of_interest, fontsize=15)
    plt.xticks(size=11)
    plt.yticks(size=15)
    plt.tight_layout()

    # Save figure
    if save_fig:
        stem = filename_stem if filename_stem else f"state_{state}_probability"
        save_path = (
            Path(config["project_path_full"]) / figures_dirname
            / f"{stem}_by_{column_of_interest}{filename_suffix}.pdf"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_zones" -- one row per session per zone, shared between
        # Plot 2 (state probability) and Plot 2b (velocity) instead of each
        # getting its own narrow sheet.
        source_data.record(state_count_df)
        source_data.flush("sessions_x_zones", figure=save_path.stem)
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
# T-Tests per genotype combo
###################################################################
def run_pairwise_ttests(
    state_count_df: pd.DataFrame,
    column_of_interest: str = "node_type",
    value_col: str = "prop",
) -> pd.DataFrame:
    """
    Perform pairwise t-tests between genotypes within each level of the column_of_interest.

    Parameters
    -----------
    state_count_df: pd.DataFrame
        DataFrame returned from compute_state_probability() (or another
        compute_*() sharing its shape, e.g. compute_zone_velocity())
    column_of_interest: str
        Column over which comparisons are grouped
    value_col : str
        Column to compare between genotypes -- "prop" (default, from
        compute_state_probability()) for HMM state proportions, or whatever
        column another compute_*() produced (e.g. "velocity").

    Returns
    --------
    pd.DataFrame
        Dataframe with columns: [Group, genotype1, genotype2, t-stat, p-value]
    """
    results = []
    groups = state_count_df[column_of_interest].dropna().unique()

    for group in groups:
        subset = state_count_df[state_count_df[column_of_interest] == group]
        genotypes = subset["genotype"].unique()

        for g1, g2 in combinations(genotypes, 2):
            values1 = subset[subset["genotype"] == g1][value_col].dropna()
            values2 = subset[subset["genotype"] == g2][value_col].dropna()
            if len(values1) >= 2 and len(values2) >= 2:
                t_stat, p_val = ttest_ind(values1, values2, equal_var=False)
                results.append({"group": group, "genotype1": g1, "genotype2": g2, "t_stat": t_stat, "p_value": p_val})

    return pd.DataFrame(results)
