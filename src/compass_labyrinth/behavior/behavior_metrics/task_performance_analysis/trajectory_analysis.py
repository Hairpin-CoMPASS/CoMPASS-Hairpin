"""
TRAJECTORY DEVIATION AND MOVEMENT DYNAMICS ACROSS BOUTS
Author: Shreya Bangera
Goal:
   ├── Deviation from Reward Path metric across bouts
   ├── Velocity across bouts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from math import ceil, sqrt

from .success_metrics import assign_bout_indices_from_entry_node, assign_bout_indices_from_entry_zone
from compass_labyrinth.constants import PATH_ORDER

import compass_labyrinth.source_data as source_data
from compass_labyrinth.utils import genotype_order as _config_genotype_order


##################################################################
# Plot 7: Deviation from Reward Path & Velocity Analysis
###################################################################
def ensure_velocity_column(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    frame_rate: float = 5.0,
) -> pd.DataFrame:
    """
    Ensure the DataFrame contains a 'Velocity' column. If not, it is computed as the Euclidean distance
    between consecutive (x, y) coordinates multiplied by the frame rate.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame.
    x_col : str
        Column name for x coordinates.
    y_col : str
        Column name for y coordinates.
    frame_rate : float
        Sampling rate (Hz) to convert framewise displacement into velocity.

    Returns
    --------
    pd.DataFrame
        Updated DataFrame with 'Velocity' column.
    """
    if "velocity" not in df.columns:
        if x_col in df.columns and y_col in df.columns:
            dx = df[x_col].diff()
            dy = df[y_col].diff()
            displacement = np.sqrt(dx**2 + dy**2)
            df["velocity"] = displacement * frame_rate
        else:
            raise ValueError(f"Missing '{x_col}' or '{y_col}' columns required to compute velocity.")
    return df


def ensure_bout_indices(df: pd.DataFrame, delimiter_node: int | list = 0) -> pd.DataFrame:
    """
    Ensure Bout indices exist.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame.
    delimiter_node : int | list
        The grid number(s) that serve as the delimiter for new bouts. Pass a
        list (e.g. constants.ENTRY_ZONE) to mark a boundary on any of
        several nodes rather than one exact node -- see
        assign_bout_indices_from_entry_zone() for why that matters when the
        entry zone is small relative to tracking sparsity.

    Returns
    --------
    pd.DataFrame
        DataFrame with ensured bout indices.
    """
    if "bout_id" not in df.columns:
        if isinstance(delimiter_node, list):
            return assign_bout_indices_from_entry_zone(df, delimiter_node)
        return assign_bout_indices_from_entry_node(df, delimiter_node)
    return df.copy()


def exp_decreasing(x, a, b, c):
    """
    Exponential decay model.
    """
    return a * np.exp(-b * x) + c


def compute_deviation_velocity(
    df: pd.DataFrame,
    key_regions: list = ["entry_zone", "reward_path", "target_zone"],
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Compute deviation and velocity per bout

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'session', 'genotype', 'region', 'grid_number', and 'velocity' columns.
    key_regions : list
        List of key regions to consider for deviation calculation.
    successful_only : bool
        If True, only include bouts that reached target_zone at least once.
        Unsuccessful bouts can never enter overshoot (target_zone is a
        prerequisite), so they contribute mechanically-forced deviation=0
        values that dilute the signal rather than reflect real behavior.

    Returns
    --------
    pd.DataFrame
        DataFrame with computed deviation and velocity per bout.
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        ind = 1
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)
        for bout in bouts_in_session:
            if len(bout) > 0:
                if successful_only and not (bout["region"] == "target_zone").any():
                    continue
                records.append(
                    {
                        "ind_no": ind,
                        "session": session_df["session"].iloc[0],
                        "genotype": session_df["genotype"].iloc[0],
                        "deviation": len(bout.loc[~bout.region.isin(key_regions), "grid_number"]) / len(bout),
                        "velocity": bout["velocity"].mean(),
                    }
                )
                ind += 1
    return pd.DataFrame(records)


def compute_deviation_velocity_by_percentage(
    df: pd.DataFrame,
    key_regions: list = ["entry_zone", "reward_path", "target_zone"],
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Like compute_deviation_velocity(), but expresses each bout's position as
    a PERCENTAGE of that session's own total (valid) bout count (0-100%)
    instead of a raw sequential bout number. Sessions have very different
    total bout counts, so a raw bout index compares "bout 50" across mice
    that may have attempted anywhere from 60 to 300 bouts total -- percentage
    instead compares "80% of the way through this mouse's own bouts" against
    the same fraction of progress for another mouse.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'session', 'genotype', 'region', 'grid_number', and 'velocity' columns.
    key_regions : list
        List of key regions to consider for deviation calculation.
    successful_only : bool
        If True, only include bouts that reached target_zone at least once
        (see compute_deviation_velocity() for why this defaults to False).

    Returns
    --------
    pd.DataFrame
        Same columns as compute_deviation_velocity(), with 'ind_no' holding
        each bout's percentage position (e.g. 33.3) instead of a raw integer
        bout count.
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)

        valid_bouts = [
            bout
            for bout in bouts_in_session
            if len(bout) > 0 and not (successful_only and not (bout["region"] == "target_zone").any())
        ]
        n_valid = len(valid_bouts)
        if n_valid == 0:
            continue

        for i, bout in enumerate(valid_bouts, start=1):
            records.append(
                {
                    "ind_no": 100 * i / n_valid,
                    "session": session_df["session"].iloc[0],
                    "genotype": session_df["genotype"].iloc[0],
                    "deviation": len(bout.loc[~bout.region.isin(key_regions), "grid_number"]) / len(bout),
                    "velocity": bout["velocity"].mean(),
                }
            )
    return pd.DataFrame(records)


def process_deviation_velocity(
    index_df: pd.DataFrame,
    genotype: str,
) -> tuple[pd.DataFrame, list, list]:
    """
    Process deviation and velocity (normalize, smooth, fit curves).

    Parameters
    -----------
    index_df : pd.DataFrame
        Input DataFrame with 'deviation', 'velocity', 'genotype', and 'ind_no' columns.
    genotype : str
        Genotype to filter the DataFrame.

    Returns
    --------
    pd.DataFrame
        Processed DataFrame with smoothed and normalized columns.
    """
    df = index_df.dropna(subset=["deviation", "velocity"])
    df = df[df["genotype"] == genotype].copy()

    # Normalize and smooth
    robust_scaler = RobustScaler()
    df["velocity_robust_scaled"] = robust_scaler.fit_transform(df[["velocity"]])

    qt = QuantileTransformer(output_distribution="uniform")
    df["velocity_normalized"] = qt.fit_transform(df[["velocity_robust_scaled"]])
    df["velocity_smooth_normalized"] = gaussian_filter1d(df["velocity_normalized"], sigma=2)
    df["deviation_smooth"] = gaussian_filter1d(df["deviation"], sigma=2)

    # Curve fitting
    x_vals = df["ind_no"].values
    y_dev = df["deviation_smooth"].values
    y_vel = df["velocity_smooth_normalized"].values

    params_dev, _ = curve_fit(exp_decreasing, x_vals, y_dev, p0=[1, 0.01, 1], maxfev=10000)
    params_vel, _ = curve_fit(exp_decreasing, x_vals, y_vel, p0=[1, 0.01, 1], maxfev=10000)

    return (df, params_dev, params_vel)


def plot_deviation_velocity_fit(
    config: dict,
    df: pd.DataFrame,
    params_dev: list,
    params_vel: list,
    genotype: str,
    max_bouts: int | None = None,
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot deviation and velocity with exponential fits for a given genotype.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    df : pd.DataFrame
        DataFrame with smoothed and normalized columns.
    params_dev : list
        Parameters for the deviation exponential fit.
    params_vel : list
        Parameters for the velocity exponential fit.
    genotype : str
        Genotype to filter the DataFrame.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) when df's 'ind_no' came from
        compute_deviation_velocity_by_percentage().
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" when
        df came from compute_deviation_velocity_by_percentage().
    title_suffix : str
        Appended to the plot title (after "... per Bout - {genotype}"), e.g.
        " (% of Bouts)" so the title itself -- not just the x-axis label --
        makes clear this is the percentage-indexed version.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    x_vals = df["ind_no"].values
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
    y_fit_dev = exp_decreasing(x_fit, *params_dev)
    y_fit_vel = exp_decreasing(x_fit, *params_vel)

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        x="ind_no",
        y="deviation_smooth",
        data=df,
        label="Smoothed Overshoot Fraction",
        linewidth=2,
        alpha=0.8,
        color="blue",
    )
    sns.lineplot(
        x="ind_no",
        y="velocity_smooth_normalized",
        data=df,
        label="Normalized Smoothed Velocity",
        linewidth=2,
        alpha=0.8,
        color="green",
    )
    plt.plot(x_fit, y_fit_dev, "--", color="red", label="Exponential Fit (Overshoot Fraction)")
    plt.plot(x_fit, y_fit_vel, "--", color="purple", label="Exponential Fit (Velocity)")
    plt.title(f"Overshoot Time Fraction and Velocity per Bout - {genotype}{title_suffix}", fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Overshoot Time Fraction / Normalized Velocity", fontsize=14)
    plt.ylim(0, 1)

    if max_bouts:
        plt.xlim(0, max_bouts)
    else:
        plt.xlim(0, df["ind_no"].max() + 5)

    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{genotype}_deviation_velocity_metric{filename_suffix}.pdf"
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
# Deviation from Reward Path and Velocity for all Genotypes
##################################################################
def plot_deviation_velocity_all(
    config: dict,
    index_df: pd.DataFrame,
    max_bouts: int | None = None,
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Creates a grid of subplots (auto-arranged) for each genotype showing smoothed
    deviation from reward path and velocity with exponential fits.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    index_df : pd.DataFrame
        DataFrame with 'deviation_smooth', 'velocity_smooth_normalized', 'genotype', and 'ind_no' columns.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) when index_df's 'ind_no' came from
        compute_deviation_velocity_by_percentage().
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" when
        index_df came from compute_deviation_velocity_by_percentage().
    title_suffix : str
        Appended to the figure's suptitle (after "... per Genotype"), e.g.
        " (% of Bouts)" so the title itself -- not just the x-axis label --
        makes clear this is the percentage-indexed version.
    filename_suffix : str
        Appended to the saved PDF's filename (before the extension), e.g.
        "_successful_only", so alternate runs (like a successful-bouts-only
        pass) don't overwrite the default saved figure.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    genotypes = sorted(index_df["genotype"].unique())
    n = len(genotypes)
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        df, params_dev, params_vel = process_deviation_velocity(index_df, genotype)
        ax = axes[i]

        # Data
        x_vals = df["ind_no"].values
        y_dev = df["deviation_smooth"].values
        y_vel = df["velocity_smooth_normalized"].values
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
        y_fit_dev = exp_decreasing(x_fit, *params_dev)
        y_fit_vel = exp_decreasing(x_fit, *params_vel)

        # Plot smoothed data
        sns.lineplot(
            x=x_vals,
            y=y_dev,
            ax=ax,
            label="Smoothed Overshoot Fraction",
            color="blue",
            linewidth=2,
        )
        sns.lineplot(
            x=x_vals,
            y=y_vel,
            ax=ax,
            label="Smoothed Velocity",
            color="green",
            linewidth=2,
        )

        # Plot exponential fits
        ax.plot(x_fit, y_fit_dev, "--", color="red", label="Exp Fit (Overshoot Fraction)")
        ax.plot(x_fit, y_fit_vel, "--", color="purple", label="Exp Fit (Velocity)")

        ax.set_title(f"{genotype}", fontsize=14, weight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Overshoot Fraction / Velocity", fontsize=12)
        if max_bouts:
            ax.set_xlim(0, max_bouts)
        else:
            ax.set_xlim(0, df["ind_no"].max() + 5)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Overshoot Time Fraction and Velocity per Genotype{title_suffix}", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_fig:
        save_path = (
            Path(config["project_path_full"]) / figures_dirname / f"all_genotypes_deviation_velocity_metric{filename_suffix}.pdf"
        )
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
# Zone-Set Occupancy and Velocity Analysis
# (generalizes Plot 7's single "deviation" (overshoot) fraction to any
# ZONE_SET -- one occupancy line per zone instead of a single in/out-of-
# key_regions fraction. Pass zone_names=["overshoot"] to reproduce the
# original Plot 7 deviation line exactly.)
##################################################################
def compute_zone_occupancy_velocity(
    df: pd.DataFrame,
    zones: dict,
    zone_names: list,
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Compute, per bout, each zone's occupancy fraction (frames in that zone /
    frames in the bout) and mean velocity -- long format, one row per
    (bout, zone) pair, so any number of zones can be plotted as separate
    lines instead of Plot 7's single "deviation" (overshoot) fraction.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'session', 'genotype', 'region', 'grid_number', and 'velocity' columns.
    zones : dict
        {zone_name: [grid_number, ...]} -- looked up per bout's grid_number
        column to compute that zone's occupancy fraction independently.
        Zones may overlap; each zone's fraction is computed on its own (a
        frame counts toward every zone it belongs to, not just one).
    zone_names : list
        Zone names (keys of `zones`) to compute -- one occupancy line per name.
    successful_only : bool
        If True, only include bouts that reached target_zone at least once
        (see compute_deviation_velocity() for why this defaults to False).

    Returns
    --------
    pd.DataFrame
        Columns: 'ind_no' (sequential bout number), 'session', 'genotype',
        'zone', 'occupancy' (fraction of that bout's frames in that zone),
        'velocity' (mean velocity for the bout, repeated per zone row).
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        ind = 1
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)
        for bout in bouts_in_session:
            if len(bout) > 0:
                if successful_only and not (bout["region"] == "target_zone").any():
                    continue
                velocity = bout["velocity"].mean()
                for zone in zone_names:
                    records.append(
                        {
                            "ind_no": ind,
                            "session": session_df["session"].iloc[0],
                            "genotype": session_df["genotype"].iloc[0],
                            "zone": zone,
                            "occupancy": bout["grid_number"].isin(zones[zone]).sum() / len(bout),
                            "velocity": velocity,
                        }
                    )
                ind += 1
    return pd.DataFrame(records)


def compute_zone_occupancy_velocity_by_percentage(
    df: pd.DataFrame,
    zones: dict,
    zone_names: list,
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Like compute_zone_occupancy_velocity(), but expresses each bout's
    position as a PERCENTAGE of that session's own total (valid) bout count
    (0-100%) instead of a raw sequential bout number (see
    compute_deviation_velocity_by_percentage() for the rationale).

    Returns
    --------
    pd.DataFrame
        Same columns as compute_zone_occupancy_velocity(), with 'ind_no'
        holding each bout's percentage position instead of a raw bout count.
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)

        valid_bouts = [
            bout
            for bout in bouts_in_session
            if len(bout) > 0 and not (successful_only and not (bout["region"] == "target_zone").any())
        ]
        n_valid = len(valid_bouts)
        if n_valid == 0:
            continue

        for i, bout in enumerate(valid_bouts, start=1):
            velocity = bout["velocity"].mean()
            for zone in zone_names:
                records.append(
                    {
                        "ind_no": 100 * i / n_valid,
                        "session": session_df["session"].iloc[0],
                        "genotype": session_df["genotype"].iloc[0],
                        "zone": zone,
                        "occupancy": bout["grid_number"].isin(zones[zone]).sum() / len(bout),
                        "velocity": velocity,
                    }
                )
    return pd.DataFrame(records)


def process_zone_occupancy_velocity(
    index_df: pd.DataFrame,
    genotype: str,
    zone_names: list,
    n_bins: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict, list]:
    """
    Process zone occupancy and velocity (bin, normalize, smooth, fit curves).

    Bins 'ind_no' into n_bins equal-width bins (bin center becomes the new
    'ind_no') so that multiple sessions' bouts share the same x-position --
    this is what lets a real cross-session 95% CI band get computed at all
    (see below), the same way process_deviation_velocity()/
    process_backtracking_velocity() get theirs "for free" from integer bout
    numbers densely overlapping across sessions.

    Unlike those two functions, this smooths per-bout values TWICE, for two
    different reasons, before anything gets aggregated across sessions:
      0. Per-session pre-smoothing: process_deviation_velocity()/
         process_backtracking_velocity() run gaussian_filter1d on the raw
         per-bout column BEFORE any cross-session aggregation happens (their
         one gaussian_filter1d call, applied to the whole session-ordered
         sequence, is what does this) -- so by the time seaborn's CI
         machinery sees the data, each observation already reflects a
         handful of that session's own neighboring bouts, not the raw,
         noisy single-bout value. Skipping this step and computing the CI
         straight from raw per-bout occupancy/velocity produces a
         cross-session variance ~1.5-2x wider than theirs, because raw
         bout-to-bout swings (e.g. a bout that grazes a zone vs. one that
         doesn't) are much larger than the smoothed signal's session-to-
         session variability. This function replicates that step explicitly
         via a per-session groupby+transform (cleaner than smoothing the
         whole concatenated dataframe as those two functions do, since it
         can't blend values across different sessions' boundaries).
      1. The mean -- one point per bin (a clean single sequence with no
         session-interleaving) of the pre-smoothed values, then
         gaussian_filter1d over the bin-mean sequence itself too. Smoothing
         raw per-bout rows directly across a WHOLE dataset (as step 0 does
         it) only works for keeping x roughly ordered because integer bout
         numbers happen to keep each session's rows roughly in x-order;
         percentage-indexed rows from different sessions interleave when
         sorted by x, so this second, bin-level smoothing pass is still
         needed on top of step 0 to keep the final plotted line itself smooth.
      2 & 3. The 95% CI band's lower and upper bounds (mean +/- 1.96*SEM,
         computed per bin from step 0's pre-smoothed per-bout values) --
         smoothed the same way as the mean, for the same reason: a
         bin-to-bin-noisy band (each bin's SEM computed from a handful of
         bouts) looks jagged right next to a smooth mean line otherwise.
         Bins with fewer than 2 observations have no defined SEM; treated as
         a zero-width band at that bin rather than dropped, so the mean
         line's resolution isn't reduced just because a local CI couldn't be
         estimated.
    The exponential fit is fit against the smoothed mean sequence (a clean,
    single point per bin) rather than the noisier per-bout values.

    Parameters
    -----------
    index_df : pd.DataFrame
        Output of compute_zone_occupancy_velocity() or its percentage variant.
    genotype : str
        Genotype to filter the DataFrame.
    zone_names : list
        Zone names (must match the 'zone' column's values) to process.
    n_bins : int
        Number of equal-width bins to divide the observed 'ind_no' range
        into -- each bout's 'ind_no' is replaced by its bin's center.

    Returns
    --------
    df_zones : pd.DataFrame
        One row per bout per zone (NOT bin-averaged), with 'ind_no' replaced
        by its bin center and the raw 'occupancy' value -- kept for any
        other per-bout diagnostics; not needed by the plotting functions.
    df_bouts : pd.DataFrame
        One row per bout (velocity is bout-level, not zone-level, so it's
        deduplicated on the original per-bout 'ind_no' before it gets
        overwritten by the shared bin center), with the raw
        'velocity_normalized' value -- same note as df_zones.
    smooth_by_zone : dict
        {zone_name: pd.DataFrame({'ind_no': bin center, 'occupancy_smooth':
        smoothed bin mean, 'ci_lower': smoothed lower 95% CI bound,
        'ci_upper': smoothed upper bound})} -- one row per (non-empty) bin.
        Plot 'occupancy_smooth' directly for the line and fill_between
        ci_lower/ci_upper for the band.
    vel_smooth : pd.DataFrame
        Same idea as smooth_by_zone's values, for the single shared velocity
        line: columns 'ind_no', 'velocity_smooth_normalized', 'ci_lower', 'ci_upper'.
    params_by_zone : dict
        {zone_name: exponential fit params}, fit against smooth_by_zone[zone].
    params_vel : list
        Exponential fit params for the (single, shared) velocity curve, fit
        against vel_smooth.
    """
    df = index_df.dropna(subset=["occupancy", "velocity"])
    df = df[df["genotype"] == genotype].copy()

    bin_edges = np.linspace(df["ind_no"].min(), df["ind_no"].max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = pd.cut(df["ind_no"], bins=bin_edges, labels=False, include_lowest=True).to_numpy()

    def _smoothed_mean_and_ci(grouped: pd.core.groupby.SeriesGroupBy, value_col: str) -> pd.DataFrame:
        stats = grouped.agg(["mean", "sem"]).reindex(range(n_bins))
        smooth = pd.DataFrame({"ind_no": bin_centers, value_col: stats["mean"].to_numpy(), "ci_half": stats["sem"].to_numpy()})
        smooth = smooth.dropna(subset=[value_col]).reset_index(drop=True)
        # Bins with < 2 raw observations have an undefined SEM (NaN) -- treat
        # as a zero-width band there instead of dropping the bin outright.
        smooth["ci_half"] = smooth["ci_half"].fillna(0) * 1.96
        smooth[value_col] = gaussian_filter1d(smooth[value_col], sigma=2)
        smooth["ci_lower"] = gaussian_filter1d(smooth[value_col] - smooth["ci_half"], sigma=2)
        smooth["ci_upper"] = gaussian_filter1d(smooth[value_col] + smooth["ci_half"], sigma=2)
        return smooth.drop(columns="ci_half")

    # Velocity is per-bout, repeated across every zone's row for that bout --
    # dedupe using the ORIGINAL fine-grained 'ind_no' (still unique per bout
    # at this point) before it gets overwritten by the shared bin center.
    keep = ~df.duplicated(subset=["session", "ind_no"])
    df_bouts = df.loc[keep].copy()
    df_bouts["bin"] = bin_idx[keep.to_numpy()]

    robust_scaler = RobustScaler()
    df_bouts["velocity_robust_scaled"] = robust_scaler.fit_transform(df_bouts[["velocity"]])
    qt = QuantileTransformer(output_distribution="uniform", n_quantiles=min(len(df_bouts), 1000))
    df_bouts["velocity_normalized"] = qt.fit_transform(df_bouts[["velocity_robust_scaled"]])

    # Pre-smooth each session's own sequence (rows are still in that
    # session's original bout order at this point) before binning -- see the
    # docstring's step 0. Rows are grouped by "session" (not resorted), so
    # each group's gaussian_filter1d only ever blends a session with itself.
    df_bouts["velocity_normalized"] = df_bouts.groupby("session")["velocity_normalized"].transform(
        lambda s: gaussian_filter1d(s.to_numpy(), sigma=2)
    )
    df_bouts["ind_no"] = bin_centers[df_bouts["bin"].to_numpy()]

    vel_smooth = _smoothed_mean_and_ci(df_bouts.groupby("bin")["velocity_normalized"], "velocity_smooth_normalized")
    params_vel, _ = curve_fit(
        exp_decreasing, vel_smooth["ind_no"].to_numpy(), vel_smooth["velocity_smooth_normalized"].to_numpy(),
        p0=[1, 0.01, 1], maxfev=10000,
    )

    df["bin"] = bin_idx
    params_by_zone = {}
    smooth_by_zone = {}
    zone_parts = []
    for zone in zone_names:
        df_z = df.loc[df["zone"] == zone].copy()
        # Same per-session pre-smoothing as velocity above, before binning.
        df_z["occupancy"] = df_z.groupby("session")["occupancy"].transform(
            lambda s: gaussian_filter1d(s.to_numpy(), sigma=2)
        )
        df_z["ind_no"] = bin_centers[df_z["bin"].to_numpy()]
        zone_parts.append(df_z)

        z_smooth = _smoothed_mean_and_ci(df_z.groupby("bin")["occupancy"], "occupancy_smooth")
        params_zone, _ = curve_fit(
            exp_decreasing, z_smooth["ind_no"].to_numpy(), z_smooth["occupancy_smooth"].to_numpy(),
            p0=[1, 0.01, 1], maxfev=10000,
        )
        params_by_zone[zone] = params_zone
        smooth_by_zone[zone] = z_smooth
    df_zones = pd.concat(zone_parts, ignore_index=True)

    return (df_zones, df_bouts, smooth_by_zone, vel_smooth, params_by_zone, params_vel)


def plot_zone_occupancy_velocity_fit(
    config: dict,
    smooth_by_zone: dict,
    vel_smooth: pd.DataFrame,
    params_by_zone: dict,
    params_vel: list,
    genotype: str,
    zone_names: list,
    max_bouts: int | None = None,
    palette: str = "tab10",
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    show_velocity: bool = True,
    ylim: tuple = (0, 1),
    occupancy_df: pd.DataFrame | None = None,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot per-zone occupancy fractions and velocity, each with an exponential
    fit, for a given genotype -- generalizes plot_deviation_velocity_fit()
    from one "deviation" line to N zone lines.

    Both the line and its 95% CI band are drawn from the SAME pre-smoothed
    sequence (smooth_by_zone/vel_smooth) -- process_zone_occupancy_velocity()
    already computed and gaussian-smoothed the mean and the CI bounds
    together, so both come out equally smooth here; there's no raw,
    per-bout data left to reconcile at plot time.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    smooth_by_zone : dict
        Output of process_zone_occupancy_velocity()'s 3rd return value
        (smoothed mean + CI bounds, for the line and band).
    vel_smooth : pd.DataFrame
        Output of process_zone_occupancy_velocity()'s 4th return value
        (smoothed mean + CI bounds, for the line and band).
    params_by_zone : dict
        Output of process_zone_occupancy_velocity()'s 5th return value.
    params_vel : list
        Output of process_zone_occupancy_velocity()'s 6th return value.
    genotype : str
        Genotype being plotted (used in the title only).
    zone_names : list
        Zone names to plot -- one color-coded line + fit per zone.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) for a percentage-indexed df.
    palette : str
        Named qualitative palette to draw one color per zone from.
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" for
        a percentage-indexed df.
    title_suffix : str
        Appended to the plot title, e.g. " (% of Bouts)".
    show_velocity : bool, default True
        Whether to draw the normalized velocity line/band/fit alongside the
        zone occupancy lines. Set False to plot zone occupancy only.
    ylim : tuple, default (0, 1)
        Y-axis limits -- zone occupancy fractions (and normalized velocity)
        are bounded in [0, 1], but real values are often much smaller (e.g.
        under 0.1), so narrow this (e.g. (0, 0.1)) to use the plot area
        better and see more detail in the lines/CI bands.
    occupancy_df : pd.DataFrame | None
        One of process_zone_occupancy_velocity()'s per-bout return values
        (df_zones or df_bouts), already filtered to this genotype -- if
        given, its unique 'session' count is appended to the title as
        "(n=<count>)". None (default) leaves the title as-is.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    colors = sns.color_palette(palette, n_colors=len(zone_names))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Built as (smoothed_handle, fit_handle) pairs, one per zone (+ velocity
    # last) -- so the legend below can lay them out two per row (smoothed
    # left, fit right, same zone) instead of leaving matplotlib's default
    # fill order to interleave them however it likes.
    legend_pairs = []
    x_max_candidates = [vel_smooth["ind_no"].max()] if show_velocity else []
    for zone, color in zip(zone_names, colors):
        z_smooth = smooth_by_zone[zone]
        x_max_candidates.append(z_smooth["ind_no"].max())

        ax.fill_between(z_smooth["ind_no"], z_smooth["ci_lower"], z_smooth["ci_upper"], color=color, alpha=0.2)
        (smoothed_line,) = ax.plot(
            z_smooth["ind_no"], z_smooth["occupancy_smooth"],
            linewidth=2, alpha=0.9, color=color, label=f"Smoothed ({zone})",
        )
        x_fit = np.linspace(z_smooth["ind_no"].min(), z_smooth["ind_no"].max(), 1000)
        y_fit = exp_decreasing(x_fit, *params_by_zone[zone])
        (fit_line,) = ax.plot(x_fit, y_fit, "--", color=color, alpha=0.6, label=f"Exponential Fit ({zone})")
        legend_pairs.append((smoothed_line, fit_line))

    if show_velocity:
        ax.fill_between(vel_smooth["ind_no"], vel_smooth["ci_lower"], vel_smooth["ci_upper"], color="black", alpha=0.15)
        (vel_line,) = ax.plot(
            vel_smooth["ind_no"], vel_smooth["velocity_smooth_normalized"],
            linewidth=2, alpha=0.9, color="black", label="Smoothed (Velocity)",
        )
        x_fit_vel = np.linspace(vel_smooth["ind_no"].min(), vel_smooth["ind_no"].max(), 1000)
        y_fit_vel = exp_decreasing(x_fit_vel, *params_vel)
        (vel_fit_line,) = ax.plot(x_fit_vel, y_fit_vel, "--", color="gray", label="Exponential Fit (Velocity)")
        legend_pairs.append((vel_line, vel_fit_line))

    # matplotlib's legend(ncol=2) fills column-major (the whole first column
    # top-to-bottom, then the second) -- so to get "left column = Smoothed,
    # right column = Exponential Fit, same row = same zone", every Smoothed
    # handle must come first (in zone order), then every Fit handle in that
    # SAME zone order, not interleaved pair by pair.
    legend_handles = [pair[0] for pair in legend_pairs] + [pair[1] for pair in legend_pairs]
    legend_labels = [h.get_label() for h in legend_handles]

    title_metric = "Zone Occupancy and Velocity" if show_velocity else "Zone Occupancy"
    n_suffix = f" (n={occupancy_df['session'].nunique()})" if occupancy_df is not None else ""
    plt.title(f"{title_metric} per Bout - {genotype}{title_suffix}{n_suffix}", fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    ylabel = "Zone Occupancy Fraction / Normalized Velocity" if show_velocity else "Zone Occupancy Fraction"
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(*ylim)

    if max_bouts:
        plt.xlim(0, max_bouts)
    else:
        plt.xlim(0, max(x_max_candidates) + 5)

    ax.legend(handles=legend_handles, labels=legend_labels, frameon=False, fontsize=9, ncol=2)
    sns.despine()
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{genotype}_zone_occupancy_velocity_metric{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # One tidy "zone" column (not one column per zone), so every genotype
        # this is called for accumulates into the same sheet.
        smoothed_long = pd.concat(
            [df.assign(zone=zone) for zone, df in smooth_by_zone.items()], ignore_index=True
        ).assign(genotype=genotype)
        source_data.record(smoothed_long)
        if show_velocity:
            source_data.record(vel_smooth.assign(genotype=genotype), part="velocity")
        source_data.flush(Path(figures_dirname).name)
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
# Zone-Set Occupancy and Velocity for all Genotypes
##################################################################
def plot_zone_occupancy_velocity_all(
    config: dict,
    index_df: pd.DataFrame,
    zone_names: list,
    max_bouts: int | None = None,
    n_bins: int = 30,
    palette: str = "tab10",
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    show_velocity: bool = True,
    ylim: tuple = (0, 1),
    genotype_order: list[str] | None = None,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Creates a grid of subplots (auto-arranged) for each genotype showing
    smoothed per-zone occupancy fractions and velocity with exponential fits
    -- generalizes plot_deviation_velocity_all() from one "deviation" line
    to N zone lines.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    index_df : pd.DataFrame
        Output of compute_zone_occupancy_velocity() or its percentage variant.
    zone_names : list
        Zone names to plot -- one color-coded line + fit per zone.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) for a percentage-indexed index_df.
    n_bins : int
        Passed through to process_zone_occupancy_velocity() -- number of
        bins spanning the observed 'ind_no' range. Higher values keep more
        of the original per-bout resolution (matching plot_deviation_velocity_all()'s
        implicit per-integer-bout-number resolution more closely) at the
        cost of fewer bouts averaged per bin (each bin's mean/CI is noisier
        before smoothing, though the line and band are gaussian-smoothed
        either way).
    palette : str
        Named qualitative palette to draw one color per zone from.
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" for
        a percentage-indexed index_df.
    title_suffix : str
        Appended to the figure's suptitle, e.g. " (% of Bouts)".
    show_velocity : bool, default True
        Whether to draw the normalized velocity line/band/fit alongside the
        zone occupancy lines. Set False to plot zone occupancy only.
    ylim : tuple, default (0, 1)
        Y-axis limits (shared across all subplots, via sharey=True) -- zone
        occupancy fractions (and normalized velocity) are bounded in [0, 1],
        but real values are often much smaller (e.g. under 0.1), so narrow
        this (e.g. (0, 0.1)) to use the plot area better and see more detail.
    genotype_order : list[str] | None
        Order to draw genotype subplots in, e.g.
        `list(GENOTYPE_SETS[GENOTYPE_SET].keys())` to match your notebook's
        own GENOTYPE_SETS definition instead of alphabetical order. Genotypes
        in index_df but not listed here are dropped; entries listed here but
        absent from index_df are skipped. Defaults to alphabetically sorted
        (index_df["genotype"].unique()) if not given.
    filename_suffix : str
        Appended to the saved PDF's filename (before the extension).
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    if genotype_order is not None:
        present = set(index_df["genotype"].unique())
        genotypes = [g for g in genotype_order if g in present]
    else:
        genotypes = _config_genotype_order(index_df["genotype"])
    n = len(genotypes)
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)
    colors = sns.color_palette(palette, n_colors=len(zone_names))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    _recorded_zones = []
    _recorded_bouts = []
    for i, genotype in enumerate(genotypes):
        df_zones, df_bouts, smooth_by_zone, vel_smooth, params_by_zone, params_vel = process_zone_occupancy_velocity(
            index_df, genotype, zone_names, n_bins=n_bins
        )
        # The raw per-session, per-bout values (not the smoothed/aggregated
        # curves plotted below) -- session-level granularity is what lets
        # the figure report's Prism tables paste this into a real Prism
        # data table with per-session replicates.
        _recorded_zones.append(df_zones.assign(genotype=genotype))
        _recorded_bouts.append(df_bouts.assign(genotype=genotype))
        ax = axes[i]

        x_max_candidates = [vel_smooth["ind_no"].max()] if show_velocity else []
        for zone, color in zip(zone_names, colors):
            z_smooth = smooth_by_zone[zone]
            x_max_candidates.append(z_smooth["ind_no"].max())

            ax.fill_between(z_smooth["ind_no"], z_smooth["ci_lower"], z_smooth["ci_upper"], color=color, alpha=0.2)
            ax.plot(z_smooth["ind_no"], z_smooth["occupancy_smooth"], label=zone, color=color, linewidth=2)
            x_fit = np.linspace(z_smooth["ind_no"].min(), z_smooth["ind_no"].max(), 1000)
            y_fit = exp_decreasing(x_fit, *params_by_zone[zone])
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.6)

        if show_velocity:
            ax.fill_between(vel_smooth["ind_no"], vel_smooth["ci_lower"], vel_smooth["ci_upper"], color="black", alpha=0.15)
            ax.plot(vel_smooth["ind_no"], vel_smooth["velocity_smooth_normalized"], label="Velocity", color="black", linewidth=2)
            x_fit_vel = np.linspace(vel_smooth["ind_no"].min(), vel_smooth["ind_no"].max(), 1000)
            y_fit_vel = exp_decreasing(x_fit_vel, *params_vel)
            ax.plot(x_fit_vel, y_fit_vel, "--", color="gray")

        n_sessions = index_df.loc[index_df["genotype"] == genotype, "session"].nunique()
        ax.set_title(f"{genotype} (n={n_sessions})", fontsize=14, weight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Zone Occupancy / Velocity" if show_velocity else "Zone Occupancy", fontsize=12)
        if max_bouts:
            ax.set_xlim(0, max_bouts)
        else:
            ax.set_xlim(0, max(x_max_candidates) + 5)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    title_metric = "Zone Occupancy and Velocity" if show_velocity else "Zone Occupancy"
    fig.suptitle(f"{title_metric} per Genotype{title_suffix}", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_fig:
        save_path = (
            Path(config["project_path_full"]) / figures_dirname / f"all_genotypes_zone_occupancy_velocity_metric{filename_suffix}.pdf"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_bouts_x_zones" -- one row per session per bout per
        # zone (plus a "successful_only" flag so this and the
        # restricted-to-successful-bouts call both accumulate into the same
        # sheet instead of separate ones), shared across Plot 6/6b instead
        # of each getting its own narrow sheet.
        successful_only = "successful_only" in filename_suffix
        # "ind_no" is a raw bout count for Plot 6 but a % of that session's
        # own bouts for Plot 6b -- both flush into this same sheet, so tag
        # which scale each row uses rather than silently mixing the two.
        is_pct = "%" in xlabel
        source_data.record(
            pd.concat(_recorded_zones, ignore_index=True).assign(successful_only=successful_only, pct_of_bouts=is_pct)
        )
        source_data.flush("sessions_x_bouts_x_zones", figure=save_path.stem)
        # Bout-level velocity (one row per session per bout, no zone) in its
        # own sheet -- dropping the zone/occupancy columns df_bouts inherits
        # from whichever zone row survived its per-bout dedup.
        source_data.record(
            pd.concat(_recorded_bouts, ignore_index=True)
            .drop(columns=["zone", "occupancy"], errors="ignore")
            .assign(successful_only=successful_only, pct_of_bouts=is_pct)
        )
        source_data.flush("sessions_x_bouts", figure=save_path.stem)
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
# Plot 7b: Corridor Backtracking & Velocity
# (adaptation of Plot 7 for a maze with no branching decision points --
# "deviation" there measured time outside the intended regions, which in
# this maze is just the overshoot fraction already covered by Plots 3-6.
# Here we measure actual backward movement along the one-way corridor,
# which is the only navigational "error" a mouse can still make: it can
# retrace its steps, but it can never take a wrong turn. path_efficiency
# (net path progress / total distance) was tried and dropped: since a bout
# always starts and ends at the entry zone by definition, its net progress
# over the full round trip is ~0 regardless of behavior, successful or not.)
##################################################################
def compute_backtracking_velocity(
    df: pd.DataFrame,
    path_order: dict = PATH_ORDER,
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Compute backtrack fraction and velocity per bout.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'session', 'genotype', 'grid_number', and 'velocity' columns.
    path_order : dict
        Mapping from grid node to its sequential position along the intended
        one-way route (constants.PATH_ORDER).
    successful_only : bool
        If True, only include bouts that reached target_zone at least once.
        Unlike deviation, backtrack_fraction is not degenerate for
        unsuccessful bouts (it's actually slightly higher there on average
        in this dataset), so this defaults to False -- use it only to
        compare against the successful-bouts-only view directly.

    Returns
    --------
    pd.DataFrame
        DataFrame with, per bout: 'backtrack_fraction' (fraction of
        frame-to-frame steps that move backward along the route -- 0 means
        the mouse never retraced its steps) and mean 'velocity'.
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        ind = 1
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)
        for bout in bouts_in_session:
            if len(bout) > 1:
                if successful_only and not (bout["region"] == "target_zone").any():
                    continue
                positions = bout["grid_number"].map(path_order).values
                steps = np.diff(positions)
                if np.abs(steps).sum() == 0:
                    continue
                records.append(
                    {
                        "ind_no": ind,
                        "session": session_df["session"].iloc[0],
                        "genotype": session_df["genotype"].iloc[0],
                        "backtrack_fraction": (steps < 0).mean(),
                        "velocity": bout["velocity"].mean(),
                    }
                )
                ind += 1
    return pd.DataFrame(records)


def compute_backtracking_velocity_by_percentage(
    df: pd.DataFrame,
    path_order: dict = PATH_ORDER,
    successful_only: bool = False,
) -> pd.DataFrame:
    """
    Like compute_backtracking_velocity(), but expresses each bout's position
    as a PERCENTAGE of that session's own total (valid) bout count (0-100%)
    instead of a raw sequential bout number. Sessions have very different
    total bout counts, so a raw bout index compares "bout 50" across mice
    that may have attempted anywhere from 60 to 300 bouts total -- percentage
    instead compares "80% of the way through this mouse's own bouts" against
    the same fraction of progress for another mouse.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'session', 'genotype', 'grid_number', and 'velocity' columns.
    path_order : dict
        Mapping from grid node to its sequential position along the intended
        one-way route (constants.PATH_ORDER).
    successful_only : bool
        If True, only include bouts that reached target_zone at least once
        (see compute_backtracking_velocity() for why this defaults to False).

    Returns
    --------
    pd.DataFrame
        Same columns as compute_backtracking_velocity(), with 'ind_no'
        holding each bout's percentage position (e.g. 33.3) instead of a
        raw integer bout count.
    """
    sessioncluster = [x for _, x in df.groupby("session")]
    records = []
    for session_df in sessioncluster:
        bouts_in_session = [x for _, x in session_df.groupby("bout_id")]
        if bouts_in_session:
            bouts_in_session.pop(0)

        valid_bouts = []
        for bout in bouts_in_session:
            if len(bout) > 1:
                if successful_only and not (bout["region"] == "target_zone").any():
                    continue
                positions = bout["grid_number"].map(path_order).values
                steps = np.diff(positions)
                if np.abs(steps).sum() == 0:
                    continue
                valid_bouts.append((bout, steps))

        n_valid = len(valid_bouts)
        if n_valid == 0:
            continue

        for i, (bout, steps) in enumerate(valid_bouts, start=1):
            records.append(
                {
                    "ind_no": 100 * i / n_valid,
                    "session": session_df["session"].iloc[0],
                    "genotype": session_df["genotype"].iloc[0],
                    "backtrack_fraction": (steps < 0).mean(),
                    "velocity": bout["velocity"].mean(),
                }
            )
    return pd.DataFrame(records)


def process_backtracking_velocity(
    index_df: pd.DataFrame,
    genotype: str,
) -> tuple[pd.DataFrame, list, list]:
    """
    Process backtrack fraction and velocity (normalize, smooth, fit curves).

    Parameters
    -----------
    index_df : pd.DataFrame
        Input DataFrame with 'backtrack_fraction', 'velocity', 'genotype', and 'ind_no' columns.
    genotype : str
        Genotype to filter the DataFrame.

    Returns
    --------
    pd.DataFrame
        Processed DataFrame with smoothed and normalized columns.
    """
    df = index_df.dropna(subset=["backtrack_fraction", "velocity"])
    df = df[df["genotype"] == genotype].copy()

    # Normalize and smooth
    robust_scaler = RobustScaler()
    df["velocity_robust_scaled"] = robust_scaler.fit_transform(df[["velocity"]])

    qt = QuantileTransformer(output_distribution="uniform")
    df["velocity_normalized"] = qt.fit_transform(df[["velocity_robust_scaled"]])
    df["velocity_smooth_normalized"] = gaussian_filter1d(df["velocity_normalized"], sigma=2)
    df["backtrack_fraction_smooth"] = gaussian_filter1d(df["backtrack_fraction"], sigma=2)

    # Curve fitting
    x_vals = df["ind_no"].values
    y_backtrack = df["backtrack_fraction_smooth"].values
    y_vel = df["velocity_smooth_normalized"].values

    params_backtrack, _ = curve_fit(exp_decreasing, x_vals, y_backtrack, p0=[1, 0.01, 1], maxfev=10000)
    params_vel, _ = curve_fit(exp_decreasing, x_vals, y_vel, p0=[1, 0.01, 1], maxfev=10000)

    return (df, params_backtrack, params_vel)


def plot_backtracking_velocity_fit(
    config: dict,
    df: pd.DataFrame,
    params_backtrack: list,
    params_vel: list,
    genotype: str,
    max_bouts: int | None = None,
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot backtrack fraction and velocity with exponential fits for a given genotype.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    df : pd.DataFrame
        DataFrame with smoothed and normalized columns.
    params_backtrack : list
        Parameters for the backtrack-fraction exponential fit.
    params_vel : list
        Parameters for the velocity exponential fit.
    genotype : str
        Genotype to filter the DataFrame.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) when df's 'ind_no' came from
        compute_backtracking_velocity_by_percentage().
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" when
        df came from compute_backtracking_velocity_by_percentage().
    title_suffix : str
        Appended to the plot title (after "... per Bout - {genotype}"), e.g.
        " (% of Bouts)" so the title itself -- not just the x-axis label --
        makes clear this is the percentage-indexed version.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    x_vals = df["ind_no"].values
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
    y_fit_backtrack = exp_decreasing(x_fit, *params_backtrack)
    y_fit_vel = exp_decreasing(x_fit, *params_vel)

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        x="ind_no", y="backtrack_fraction_smooth", data=df,
        label="Smoothed Backtrack Fraction", linewidth=2, alpha=0.8, color="blue",
    )
    sns.lineplot(
        x="ind_no", y="velocity_smooth_normalized", data=df,
        label="Normalized Smoothed Velocity", linewidth=2, alpha=0.8, color="green",
    )
    plt.plot(x_fit, y_fit_backtrack, "--", color="red", label="Exponential Fit (Backtrack Fraction)")
    plt.plot(x_fit, y_fit_vel, "--", color="purple", label="Exponential Fit (Velocity)")
    plt.title(f"Corridor Backtrack Fraction and Velocity per Bout - {genotype}{title_suffix}", fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Backtrack Fraction / Normalized Velocity", fontsize=14)
    plt.ylim(0, 1)

    if max_bouts:
        plt.xlim(0, max_bouts)
    else:
        plt.xlim(0, df["ind_no"].max() + 5)

    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{genotype}_backtracking_velocity_metric{filename_suffix}.pdf"
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
# Corridor Backtracking and Velocity for all Genotypes
##################################################################
def plot_backtracking_velocity_all(
    config: dict,
    index_df: pd.DataFrame,
    max_bouts: int | None = None,
    xlabel: str = "Bout Number",
    title_suffix: str = "",
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Creates a grid of subplots (auto-arranged) for each genotype showing smoothed
    backtrack fraction and velocity with exponential fits.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    index_df : pd.DataFrame
        DataFrame with 'backtrack_fraction', 'velocity', 'genotype', and 'ind_no' columns.
    max_bouts : int or None
        Maximum value to display on the x-axis -- a raw bout count, or a max
        percentage (e.g. 100) when index_df's 'ind_no' came from
        compute_backtracking_velocity_by_percentage().
    xlabel : str
        X-axis label -- "Bout Number" by default, pass e.g. "% of Bouts" when
        index_df came from compute_backtracking_velocity_by_percentage().
    title_suffix : str
        Appended to the figure's suptitle (after "... per Genotype"), e.g.
        " (% of Bouts)" so the title itself -- not just the x-axis label --
        makes clear this is the percentage-indexed version.
    filename_suffix : str
        Appended to the saved PDF's filename (before the extension), e.g.
        "_successful_only", so alternate runs don't overwrite the default
        saved figure.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    genotypes = sorted(index_df["genotype"].unique())
    n = len(genotypes)
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        df, params_backtrack, params_vel = process_backtracking_velocity(index_df, genotype)
        ax = axes[i]

        # Data
        x_vals = df["ind_no"].values
        y_backtrack = df["backtrack_fraction_smooth"].values
        y_vel = df["velocity_smooth_normalized"].values
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
        y_fit_backtrack = exp_decreasing(x_fit, *params_backtrack)
        y_fit_vel = exp_decreasing(x_fit, *params_vel)

        # Plot smoothed data
        sns.lineplot(x=x_vals, y=y_backtrack, ax=ax, label="Backtrack Fraction", color="blue", linewidth=2)
        sns.lineplot(x=x_vals, y=y_vel, ax=ax, label="Velocity", color="green", linewidth=2)

        # Plot exponential fits
        ax.plot(x_fit, y_fit_backtrack, "--", color="red", label="Exp Fit (Backtrack Fraction)")
        ax.plot(x_fit, y_fit_vel, "--", color="purple", label="Exp Fit (Velocity)")

        ax.set_title(f"{genotype}", fontsize=14, weight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Backtrack Fraction / Velocity", fontsize=12)
        if max_bouts:
            ax.set_xlim(0, max_bouts)
        else:
            ax.set_xlim(0, df["ind_no"].max() + 5)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Corridor Backtrack Fraction and Velocity per Genotype{title_suffix}", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_fig:
        save_path = (
            Path(config["project_path_full"])
            / figures_dirname
            / f"all_genotypes_backtracking_velocity_metric{filename_suffix}.pdf"
        )
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
