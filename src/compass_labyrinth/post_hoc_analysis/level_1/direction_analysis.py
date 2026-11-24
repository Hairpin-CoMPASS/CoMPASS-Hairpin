"""
BOUT DEPTH AND DIRECTION OF TRAVEL (10x10 hairpin maze)
Goal:
    ├── Position along the single maze path for every frame (entry -> terminal).
    ├── Direction of travel per frame (forward = toward the reward/terminal).
    ├── Each bout's deepest point, and a phase label for every frame:
    │     outbound / backtrack (before the deepest point), return / re-entry (after it).
    └── How deep bouts go, and how often mice move against their bout's phase, by genotype.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from compass_labyrinth.utils import genotype_order, group_sample_sizes, append_n_to_legend
from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import DIFF_CMAP, hold_draws, release_draws

import compass_labyrinth.source_data as source_data


# The maze is one path with no branches: lane 0 runs entry (cells 0/1) -> right,
# a U-turn leads into lane 1 running back, and so on, alternating sides; the
# reward (70/71/80/81) is the U-turn after lane 7, and lanes 8-9 are the
# overshoot ending at the terminal (90/91). A cell's index here is its
# position along that path. (Checked on this dataset: 97% of frame-to-frame
# cell changes move 1-2 positions, 2 being a U-turn diagonal.)
FORWARD_PATH = [
    row * 10 + (col if row % 2 == 0 else 9 - col)
    for row in range(10)
    for col in range(10)
]
PATH_POSITION = {cell: pos for pos, cell in enumerate(FORWARD_PATH)}
REWARD_POSITION = min(PATH_POSITION[c] for c in (70, 71, 80, 81))  # first reward cell along the path
OVERSHOOT_POSITION = max(PATH_POSITION[c] for c in (70, 71, 80, 81)) + 1  # first overshoot cell (82)

PHASES = ["outbound", "backtrack", "at_reward", "overshoot_forward", "overshoot_backward", "return", "reentry",
          "undetermined"]
PHASE_LABELS = {
    "outbound": "Outbound (forward on the path, before the turning point)",
    "backtrack": "Backtrack (backward on the path, before the turning point)",
    "at_reward": "At reward (in the reward zone after the first arrival; reward-reaching bouts only)",
    "overshoot_forward": "Overshoot forward (deeper into the overshoot; reward-reaching bouts only)",
    "overshoot_backward": "Overshoot backward (back toward the reward in the overshoot; reward-reaching bouts only)",
    "return": "Return (backward on the path, after the turning point)",
    "reentry": "Re-entry (forward on the path, after the turning point)",
    "undetermined": "Undetermined (a bout with no movement along the path at all)",
}


def _bout_direction(pos: np.ndarray, window: int, max_step: int) -> np.ndarray:
    """+1 forward / -1 backward / 0 undetermined per frame: the sign of the net
    path change from `window` frames before to `window` frames after (clipped
    at the bout's edges), with tracking jumps of more than `max_step`
    positions ignored."""
    step = np.diff(pos, prepend=pos[0]).astype(float)
    step[np.abs(step) > max_step] = 0
    track = np.cumsum(step)
    idx = np.arange(len(pos))
    ahead = track[np.minimum(idx + window, len(pos) - 1)]
    behind = track[np.maximum(idx - window, 0)]
    return np.sign(ahead - behind).astype(int)


def _fill_pauses(direction: np.ndarray) -> np.ndarray:
    """Frames with no net movement (0) take the direction of the NEARER moving
    frame in time -- a pause splits at its midpoint, the first half going with
    the movement before it and the second half with the movement after it
    (ties go to the movement before). A bout with no movement at all stays 0."""
    idx = np.arange(len(direction))
    moving = direction != 0
    if not moving.any():
        return direction.copy()
    prev_i = pd.Series(np.where(moving, idx, np.nan)).ffill().to_numpy()
    next_i = pd.Series(np.where(moving, idx, np.nan)).bfill().to_numpy()
    use_prev = ~np.isnan(prev_i) & (np.isnan(next_i) | (idx - prev_i <= next_i - idx))
    src = np.where(use_prev, prev_i, next_i).astype(int)
    return np.where(moving, direction, direction[src])


def label_bout_directions(
    df: pd.DataFrame,
    window: int = 10,
    max_step: int = 3,
    min_bout_frames: int = 20,
) -> pd.DataFrame:
    """
    Per-frame path position, direction and bout phase.

    Parameters
    -----------
    df : pd.DataFrame
        Frames with 'session', 'genotype', 'grid_number' and 'bout_id' (from
        assign_bout_indices_from_entry_zone(); bout_id 0 = frames in the entry zone).
    window : int
        Frames before/after used for direction (10 = +-2 s at 5 fps).
    max_step : int
        Frame-to-frame path changes larger than this are tracking jumps and ignored.
    min_bout_frames : int
        Shorter bouts are dropped.

    Returns
    --------
    pd.DataFrame
        The bout frames (entry-zone frames excluded) plus 'path_pos',
        'moving' (net movement along the path within the window), 'direction'
        (+1/-1; a frame with no net movement takes the direction of the nearer
        moving frame in time, so a pause splits at its midpoint), 'deepest_pos',
        'complete' (the bout returned to the entry zone, i.e. isn't cut off by
        the session ending) and 'phase'.
        A bout that reaches the reward turns at its first arrival there:
        before it, forward = outbound, backward = backtrack; after it, by zone --
        reward zone = at_reward; overshoot: forward = overshoot_forward,
        backward = overshoot_backward; path: backward = return, forward = re-entry.
        A bout that doesn't turns at its furthest point -- the stay at the
        deepest position from its first arrival, split at its midpoint (first
        half outbound, second half return): before it, forward = outbound and
        backward = backtrack; after it, backward = return and forward = re-entry.
        ("undetermined" only for a bout with no movement at all.)
    """
    df = df[df["bout_id"] > 0].copy()
    df["path_pos"] = df["grid_number"].astype(int).map(PATH_POSITION)
    last_bout = df.groupby("session")["bout_id"].transform("max")
    df["complete"] = df["bout_id"] < last_bout
    parts = []
    for _, bout in df.groupby(["session", "bout_id"], sort=False):
        if len(bout) < min_bout_frames:
            continue
        pos = bout["path_pos"].to_numpy()
        raw = _bout_direction(pos, window, max_step)
        direction = _fill_pauses(raw)
        # Turnaround = the stay at the deepest position from its first
        # arrival; it is split at its midpoint -- first half outbound, second
        # half return -- and that midpoint is the bout's outbound/return boundary
        k = int(np.argmax(pos))
        idx = np.arange(len(pos))
        fwd = direction > 0
        if pos[k] >= REWARD_POSITION:
            # Reached the reward: its first arrival splits the bout; after it, label by zone
            after = idx >= int(np.argmax(pos >= REWARD_POSITION))
            phase = np.select(
                [direction == 0, ~after, pos >= OVERSHOOT_POSITION, pos >= REWARD_POSITION],
                ["undetermined", np.where(fwd, "outbound", "backtrack"),
                 np.where(fwd, "overshoot_forward", "overshoot_backward"), "at_reward"],
                np.where(fwd, "reentry", "return"))
        else:
            j = k
            while j + 1 < len(pos) and pos[j + 1] == pos[k]:
                j += 1
            mid = (k + j) // 2
            before = idx <= mid
            phase = np.where(
                direction == 0, "undetermined",
                np.where(before, np.where(fwd, "outbound", "backtrack"),
                         np.where(direction < 0, "return", "reentry")),
            )
            phase = np.where((idx >= k) & (idx <= mid), "outbound", phase)
            phase = np.where((idx > mid) & (idx <= j), "return", phase)
        parts.append(bout.assign(moving=raw != 0, direction=direction, deepest_pos=pos[k], phase=phase))
    # The input's row index is kept, so labels join back to the full frame table
    return pd.concat(parts) if parts else df.iloc[0:0]


def frame_phases(df: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    """Every frame of `df` (the full frame table label_bout_directions() was
    run on) with its 'bout_id', 'path_pos', 'direction', 'deepest_pos' and
    'phase'. Frames outside a labeled bout get phase "entry_zone" (in the
    entry zone between bouts) or "short_bout" (bouts under min_bout_frames)."""
    out = df.copy()
    out["path_pos"] = out["grid_number"].astype(int).map(PATH_POSITION)
    for col in ("moving", "direction", "deepest_pos", "phase"):
        out[col] = labeled[col].reindex(out.index)
    out["phase"] = out["phase"].fillna(pd.Series(np.where(out["bout_id"] == 0, "entry_zone", "short_bout"),
                                                 index=out.index))
    return out


def summarize_bouts(labeled: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """One row per bout: depth (deepest path position, its cell and lane,
    whether it reached the reward) and how many frames fell in each phase.
    complete_only drops each session's last bout when the session ended
    before the mouse came back to the entry zone."""
    if complete_only:
        labeled = labeled[labeled["complete"]]
    counts = labeled.pivot_table(index=["session", "genotype", "bout_id"], columns="phase",
                                 values="path_pos", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=PHASES, fill_value=0)
    bouts = labeled.groupby(["session", "genotype", "bout_id"]).agg(
        deepest_pos=("deepest_pos", "first"), frames=("path_pos", "size"))
    bouts = bouts.join(counts).reset_index()
    bouts["deepest_cell"] = bouts["deepest_pos"].map(lambda p: FORWARD_PATH[int(p)])
    bouts["deepest_lane"] = (bouts["deepest_pos"] // 10).astype(int)
    bouts["reached_reward"] = bouts["deepest_pos"] >= REWARD_POSITION
    return bouts


def bout_depth_by_session(bouts: pd.DataFrame) -> pd.DataFrame:
    """% of each session's bouts whose deepest point is in each lane (0-9),
    one row per session x lane (lanes a session never turned around in are 0)."""
    counts = bouts.groupby(["session", "genotype", "deepest_lane"]).size().rename("n").reset_index()
    sessions = bouts[["session", "genotype"]].drop_duplicates()
    full = sessions.merge(pd.DataFrame({"deepest_lane": range(10)}), how="cross")
    full = full.merge(counts, on=["session", "genotype", "deepest_lane"], how="left").fillna({"n": 0})
    full["pct_bouts"] = 100 * full["n"] / full.groupby("session")["n"].transform("sum")
    return full


def phase_by_session(bouts: pd.DataFrame) -> pd.DataFrame:
    """Per session: % of frames before the deepest point that are backtracking
    (backtrack / (outbound + backtrack)), % after it that are re-entering
    (reentry / (return + reentry)), and % of bouts reaching the reward."""
    s = bouts.groupby(["session", "genotype"])[PHASES + ["reached_reward"]].sum().reset_index()
    n = bouts.groupby("session").size().rename("bouts")
    s = s.merge(n, on="session")
    s["pct_backtrack"] = 100 * s["backtrack"] / (s["outbound"] + s["backtrack"]).replace(0, np.nan)
    s["pct_reentry"] = 100 * s["reentry"] / (s["return"] + s["reentry"]).replace(0, np.nan)
    s["pct_reached_reward"] = 100 * s["reached_reward"] / s["bouts"]
    return s


def depth_table(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Lane x genotype table: mean (SE) % of bouts turning around in each lane,
    each session counted once -- the readable summary of bout_depth_by_session()."""
    order = genotype_order(depth_df["genotype"])
    stats = depth_df.groupby(["deepest_lane", "genotype"])["pct_bouts"].agg(["mean", "sem"]).reset_index()
    stats["value"] = stats.apply(lambda r: f"{r['mean']:.1f} ({r['sem']:.1f})", axis=1)
    table = stats.pivot(index="deepest_lane", columns="genotype", values="value").reindex(columns=order)
    table.index = [f"lane {i} (cells {FORWARD_PATH[i * 10]}-{FORWARD_PATH[i * 10 + 9]})" for i in table.index]
    table.index.name = "deepest point"
    return table.reset_index()


def phase_table(phase_df: pd.DataFrame) -> pd.DataFrame:
    """Genotype x measure table: mean (SE) across sessions of % bouts reaching
    the reward (backtrack / re-entry are in Plot 5b, by bout type)."""
    order = genotype_order(phase_df["genotype"])
    rows = []
    for col, label in [("pct_reached_reward", "% bouts reaching the reward")]:
        row = {"measure": label}
        for g in order:
            v = phase_df.loc[phase_df["genotype"] == g, col].dropna()
            row[g] = f"{v.mean():.1f} ({v.sem():.1f})" if len(v) else ""
        rows.append(row)
    return pd.DataFrame(rows)


MOVING_PHASES = ["outbound", "backtrack", "overshoot_forward", "overshoot_backward", "return", "reentry"]
# Plot 5b's fractions: every labeled frame of a bout (sum to 1)
FRACTION_PHASES = ["outbound", "backtrack", "at_reward", "overshoot_forward", "overshoot_backward", "return", "reentry"]
PHASE_SHORT = {"outbound": "Outbound", "backtrack": "Backtrack", "at_reward": "At reward",
               "overshoot_forward": "Overshoot\nforward", "overshoot_backward": "Overshoot\nbackward",
               "return": "Return", "reentry": "Re-entry"}
PHASE_COLORS = {"outbound": "#2b8cbe", "backtrack": "#a6bddb", "at_reward": "#fec44f", "overshoot_forward": "#ef3b2c",
                "overshoot_backward": "#fc9272", "return": "#4d4d4d", "reentry": "#bdbdbd"}
BOUT_TYPES = {True: "Reached reward", False: "Did not reach reward"}


def phase_fractions_by_session(bouts: pd.DataFrame) -> pd.DataFrame:
    """Per session x bout type (reached the reward or not) x phase: the
    fraction of those bouts' frames in each of outbound / backtrack / return /
    re-entry -- the 4 fractions sum to 1 per session and bout type. Long
    format; a session with no bouts of a type has no rows for it."""
    g = bouts.groupby(["session", "genotype", "reached_reward"])[FRACTION_PHASES].sum()
    g = g[g.sum(axis=1) > 0]
    frac = g.div(g.sum(axis=1), axis=0).reset_index()
    frac["bout_type"] = frac["reached_reward"].map(BOUT_TYPES)
    long = frac.melt(id_vars=["session", "genotype", "bout_type"], value_vars=FRACTION_PHASES,
                     var_name="phase", value_name="fraction")
    n = bouts.groupby(["session", "reached_reward"]).size().rename("bouts").reset_index()
    n["bout_type"] = n["reached_reward"].map(BOUT_TYPES)
    return long.merge(n[["session", "bout_type", "bouts"]], on=["session", "bout_type"], how="left")


def phase_fraction_ttests(frac_df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests between every pair of genotype groups for each phase,
    within each bout type, with Benjamini-Hochberg FDR across the 4 phases
    (per bout type and pair)."""
    from itertools import combinations
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    rows = []
    order = genotype_order(frac_df["genotype"])
    for bout_type, sub in frac_df.groupby("bout_type", sort=False):
        for g1, g2 in combinations(order, 2):
            block = []
            for phase in [p for p in FRACTION_PHASES if sub.loc[sub["phase"] == p, "fraction"].gt(0).any()]:
                a = sub.loc[(sub["genotype"] == g1) & (sub["phase"] == phase), "fraction"].dropna()
                b = sub.loc[(sub["genotype"] == g2) & (sub["phase"] == phase), "fraction"].dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                t, p = ttest_ind(a, b, equal_var=False)
                block.append({"bout_type": bout_type, "phase": phase, "group1": g1, "group2": g2,
                              "mean1": a.mean(), "mean2": b.mean(), "n1": len(a), "n2": len(b),
                              "t_stat": t, "p_value": p})
            if block:
                p_fdr = multipletests([r["p_value"] for r in block], method="fdr_bh")[1]
                for r, q in zip(block, p_fdr):
                    r["p_fdr"] = q
                rows += block
    return pd.DataFrame(rows)


def plot_phase_fractions(
    config: dict,
    frac_df: pd.DataFrame,
    bout_type: str,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """For one bout type ("Reached reward" / "Did not reach reward"): per
    genotype group, 4 bars -- the fraction of those bouts' frames that are
    outbound, backtrack, return and re-entry (mean +- SE across sessions;
    each session's 4 fractions sum to 1)."""
    df = frac_df[frac_df["bout_type"] == bout_type].copy()
    order = genotype_order(df["genotype"])
    n = group_sample_sizes(df, "genotype")
    shown = [p for p in FRACTION_PHASES if df.loc[df["phase"] == p, "fraction"].gt(0).any()]
    df = df[df["phase"].isin(shown)]
    df["phase_label"] = df["phase"].map(PHASE_SHORT)
    fig, ax = plt.subplots(figsize=(3.2 + 0.6 * len(shown) * len(order), 5))
    sns.barplot(data=df, x="genotype", y="fraction", order=order, hue="phase_label",
                hue_order=[PHASE_SHORT[p] for p in shown],
                palette=[PHASE_COLORS[p] for p in shown], errorbar="se", capsize=0.08,
                err_kws={"linewidth": 1.2}, edgecolor="black", ax=ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(n={n.get(g, 0)})" for g in order], fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of bout frames", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(f"Time by direction -- bouts that {'reached' if bout_type == BOUT_TYPES[True] else 'did not reach'} the reward",
                 fontsize=13, weight="bold")
    ax.legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    fig.tight_layout()

    if save_fig:
        tag = "successful" if bout_type == BOUT_TYPES[True] else "unsuccessful"  # sorts successful first
        save_path = Path(config["project_path_full"]) / figures_dirname / f"phase_fractions_{tag}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_bout_type_x_phases": one row per session x bout type x phase
        source_data.record(df.drop(columns="phase_label"))
        source_data.flush("sessions_x_bout_type_x_phases", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def plot_bout_depth(
    config: dict,
    depth_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Left: % of bouts turning around in each lane (mean +- SE across
    sessions), by genotype. Right: % of bouts reaching the reward, by genotype."""
    order = genotype_order(depth_df["genotype"])
    n = group_sample_sizes(depth_df, "genotype")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 0.55]})

    sns.barplot(data=depth_df, x="deepest_lane", y="pct_bouts", hue="genotype", hue_order=order,
                errorbar="se", capsize=0.1, err_kws={"linewidth": 1.2}, edgecolor="black", ax=ax1)
    ax1.set_xlabel("Deepest lane reached in the bout (0 = entry lane, 7 = reward, 9 = terminal)", fontsize=11)
    ax1.set_ylabel("% of bouts", fontsize=12)
    ax1.set_title("How far bouts go before turning back", fontsize=13, weight="bold")
    ax1.axvspan(6.5, 7.5, color="gold", alpha=0.15, zorder=0)
    append_n_to_legend(ax1.get_legend(), n)
    ax1.get_legend().set_title("")

    long = phase_df.assign(measure="Reached reward")
    sns.barplot(data=long, x="measure", y="pct_reached_reward", hue="genotype", hue_order=order, errorbar="se",
                capsize=0.1, err_kws={"linewidth": 1.2}, edgecolor="black", ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("% of bouts", fontsize=12)
    ax2.set_title("Reward reached", fontsize=13, weight="bold")
    ax2.get_legend().remove()
    fig.tight_layout()

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"bout_depth_direction{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_lanes": one row per session x deepest lane;
        # "sessions": per-session phase rates
        source_data.record(depth_df)
        source_data.flush("sessions_x_lanes", figure=save_path.stem)
        source_data.record(phase_df)
        source_data.flush("sessions", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


##################################################################
# Plot 5d: how far backtracks and re-entries go
##################################################################
REVERSALS = {"backtrack": "Backtrack (backing up on the way in)", "reentry": "Re-entry (heading in again on the way out)"}
# cells moved -> distance class; distances are bimodal (many 1-cell moves, a group of lane-long ones)
DISTANCE_BINS = ([0, 1, 4, 9, np.inf], ["1", "2-4", "5-9", "10+"])


def reversal_occurrences(labeled: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """One row per backtrack / re-entry occurrence: a run of frames moving
    the same way (pauses included, as in Plot 5) that starts in that phase --
    so a re-entry that carries on into the reward and the overshoot lasts
    until the mouse turns again: 'frames',
    'cells' (path positions covered -- the range from the frame before the
    run through the run), where it turned around -- 'start_pos' / 'start_lane'
    (the deepest point it backed up from, for a backtrack; the point nearest
    the entry it headed in again from, for a re-entry) -- and where it ended
    up -- 'end_pos' / 'end_lane' (the point nearest the entry it backed up
    to; the deepest point it went back in to). Occurrences with cells == 0 never left their cell
    (direction is judged over +-window frames). `labeled` from
    label_bout_directions(); complete_only as in summarize_bouts()."""
    if complete_only:
        labeled = labeled[labeled["complete"]]
    d = labeled.sort_values(["session", "s_no"])
    d = d.assign(prev_pos=d.groupby(["session", "bout_id"])["path_pos"].shift())
    sign = np.sign(d["direction"])
    new_run = ((sign != sign.shift()) | (d["bout_id"] != d["bout_id"].shift())
               | (d["session"] != d["session"].shift()))
    d = d.assign(run=new_run.cumsum())
    # Keep each run's own kind of frames (a re-entry also its reward / overshoot-forward continuation),
    # so a tracking jump inside a backward run can't carry a backtrack past the reward
    allowed = {"backtrack": {"backtrack"}, "reentry": {"reentry", "at_reward", "overshoot_forward"}}
    first_phase = d.groupby("run")["phase"].transform("first")
    d = d[[ph in allowed.get(fp, {fp}) for ph, fp in zip(d["phase"], first_phase)]]
    runs = d.groupby("run").agg(session=("session", "first"), genotype=("genotype", "first"),
                                bout_id=("bout_id", "first"), phase=("phase", "first"), frames=("s_no", "size"),
                                prev_pos=("prev_pos", "first"), min_pos=("path_pos", "min"),
                                max_pos=("path_pos", "max"))
    runs = runs[runs["phase"].isin(list(REVERSALS))]
    lo = runs[["min_pos", "prev_pos"]].min(axis=1)
    hi = runs[["max_pos", "prev_pos"]].max(axis=1)
    runs["cells"] = (hi - lo).astype(int)
    back = runs["phase"] == "backtrack"
    runs["start_pos"] = np.where(back, hi, lo).astype(int)
    runs["end_pos"] = np.where(back, runs["min_pos"], runs["max_pos"]).astype(int)
    runs["start_lane"] = runs["start_pos"] // 10
    runs["end_lane"] = runs["end_pos"] // 10
    return runs.drop(columns=["prev_pos", "min_pos", "max_pos"]).reset_index(drop=True)


def _reversal_axes(figsize: tuple) -> tuple:
    """A figure with, per reversal (REVERSALS), a row of 3 panels then a row of
    2 wider ones. Returns (fig, {phase: [5 axes]})."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2 * len(REVERSALS), 6, figure=fig, hspace=0.55, wspace=0.6)
    axes = {}
    for i, phase in enumerate(REVERSALS):
        top, bottom = 2 * i, 2 * i + 1
        axes[phase] = [fig.add_subplot(gs[top, 0:2]), fig.add_subplot(gs[top, 2:4]), fig.add_subplot(gs[top, 4:6]),
                       fig.add_subplot(gs[bottom, 0:3]), fig.add_subplot(gs[bottom, 3:6])]
    return fig, axes


def _pct_by_class(e: pd.DataFrame, col: str, classes: list) -> pd.DataFrame:
    """Per session x phase x class of `col`: 'pct_occurrences' (% of that
    session's occurrences of that phase); sessions with no occurrence left out."""
    counts = e.groupby(["session", "genotype", "phase", col], observed=True).size().rename("n").reset_index()
    sp = counts[["session", "genotype", "phase"]].drop_duplicates()
    full = sp.merge(pd.DataFrame({col: classes}), how="cross")
    full = full.merge(counts, on=["session", "genotype", "phase", col], how="left").fillna({"n": 0})
    full["pct_occurrences"] = 100 * full["n"] / full.groupby(["session", "phase"])["n"].transform("sum")
    return full


def reversals_by_session(occurrences: pd.DataFrame, bouts: pd.DataFrame, min_cells: int = 1) -> tuple:
    """From reversal_occurrences() (only occurrences covering >= min_cells cells)
    and summarize_bouts(): (per session x phase -- 'occurrences_per_bout',
    'mean_cells' and 'median_cells'; per session x phase x DISTANCE_BINS class -- 'pct_occurrences'
    covering that many cells; per session x phase x 'point' (start = where it
    turned around, end = where it ended up) x 'lane' 0-9 -- 'pct_occurrences'
    there). Sessions with no such occurrence are left out of the last two."""
    e = occurrences[occurrences["cells"] >= min_cells]
    n_bouts = bouts.groupby(["session", "genotype"]).size().rename("bouts")
    full = pd.MultiIndex.from_product([n_bouts.index.get_level_values("session"), list(REVERSALS)],
                                      names=["session", "phase"])
    per = (e.groupby(["session", "phase"]).agg(occurrences=("cells", "size"), mean_cells=("cells", "mean"),
                                                median_cells=("cells", "median"))
           .reindex(full).reset_index().fillna({"occurrences": 0}))
    per = per.merge(n_bouts.reset_index(), on="session")
    per["occurrences_per_bout"] = per["occurrences"] / per["bouts"]
    edges, labels = DISTANCE_BINS
    distances = _pct_by_class(e.assign(distance=pd.cut(e["cells"], edges, labels=labels).astype(str)),
                              "distance", labels)
    lanes = pd.concat([_pct_by_class(e.rename(columns={f"{pt}_lane": "lane"}), "lane", list(range(10))).assign(point=pt)
                       for pt in ("start", "end")], ignore_index=True)
    per = per[["session", "genotype", "phase", "bouts", "occurrences", "occurrences_per_bout", "mean_cells",
               "median_cells"]]
    return per, distances, lanes


def plot_reversal_distance(
    config: dict,
    per_session: pd.DataFrame,
    distances: pd.DataFrame,
    lanes: pd.DataFrame,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Per reversal (backtrack, re-entry), a row of how often per bout (dots =
    mice), the mean cells moved (dots = mice) and % of occurrences covering
    1 / 2-4 / 5-9 / 10+ cells, then a row of % turning around in / ending up
    in each lane -- bars mean +- SE across sessions. Inputs from
    reversals_by_session()."""
    order = genotype_order(per_session["genotype"])
    n = group_sample_sizes(per_session, "genotype")
    colors = dict(zip(order, sns.color_palette(n_colors=len(order))))
    fig, axes = _reversal_axes((21, 19))
    for phase, title in REVERSALS.items():
        ax_n, ax_m, ax_d, ax_s, ax_e = axes[phase]
        p = per_session[per_session["phase"] == phase]
        for ax, col, ylabel in [(ax_n, "occurrences_per_bout", "Occurrences per bout"),
                                (ax_m, "mean_cells", "Mean cells moved (per-mouse mean)")]:
            kw = dict(data=p, x="genotype", y=col, order=order, hue="genotype", hue_order=order, palette=colors,
                      ax=ax, legend=False)
            sns.barplot(**kw, errorbar="se", capsize=0.15, err_kws={"linewidth": 1.2}, edgecolor="black", alpha=0.85)
            sns.stripplot(**kw, size=4, edgecolor="black", linewidth=0.5)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.tick_params(axis="x", labelsize=9)
        sns.barplot(data=distances[distances["phase"] == phase], x="distance", y="pct_occurrences",
                    order=DISTANCE_BINS[1], hue="genotype", hue_order=order, palette=colors, errorbar="se",
                    capsize=0.1, err_kws={"linewidth": 1.1}, edgecolor="black", ax=ax_d)
        ax_d.set_xlabel("Cells moved (10 = one lane)", fontsize=11)
        ax_d.set_ylabel("% of occurrences", fontsize=11)
        ax_d.get_legend().remove()
        ax_e.sharey(ax_s)
        for ax, pt, what in [(ax_s, "start", "turned around"), (ax_e, "end", "ended up")]:
            sns.barplot(data=lanes[(lanes["phase"] == phase) & (lanes["point"] == pt)], x="lane",
                        y="pct_occurrences", hue="genotype", hue_order=order, palette=colors, errorbar="se",
                        capsize=0.1, err_kws={"linewidth": 1.1}, edgecolor="black", ax=ax)
            ax.axvspan(6.5, 7.5, color="gold", alpha=0.15, zorder=0)
            ax.set_title(f"Where it {what}", fontsize=12)
            ax.set_xlabel("Lane (0 = entry lane, 7 = reward, 9 = terminal)", fontsize=11)
            ax.set_ylabel("% of occurrences", fontsize=11)
            ax.get_legend().remove()
        ax_n.annotate(title, xy=(0, 1.1), xycoords="axes fraction", fontsize=13, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[g], edgecolor="black") for g in order]
    legend = fig.legend(handles, order, loc="lower center", ncol=len(order), frameon=False, fontsize=10)
    append_n_to_legend(legend, n)
    fig.suptitle("How far backtracks and re-entries go", fontsize=14, weight="bold", y=0.93)

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"reversal_distance{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_reversals": per session x phase; "_distances" / "_lanes": per session x phase x class
        source_data.record(per_session)
        source_data.record(distances, part="distances")
        source_data.record(lanes, part="lanes")
        source_data.flush("sessions_x_reversals", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def reversal_distance_ttests(per_session: pd.DataFrame, distances: pd.DataFrame,
                             lanes: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests between every pair of genotype groups, per reversal
    (backtrack, re-entry): occurrences per bout, % of occurrences in each distance
    class and % turning around in / ending up in each lane; Benjamini-Hochberg FDR across
    the table's rows within each reversal x pair."""
    from itertools import combinations
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    long = pd.concat([
        per_session.melt(id_vars=["session", "genotype", "phase"], value_vars=["occurrences_per_bout", "mean_cells"],
                         var_name="measure"),
        distances.assign(measure="% moving " + distances["distance"] + " cells",
                         value=distances["pct_occurrences"])[["session", "genotype", "phase", "measure", "value"]],
        lanes.assign(measure=lanes["point"].map({"start": "% turning around in lane ", "end": "% ending up in lane "})
                     + lanes["lane"].astype(str),
                     value=lanes["pct_occurrences"])[["session", "genotype", "phase", "measure", "value"]],
    ])
    order = genotype_order(long["genotype"])
    rows = []
    for phase in REVERSALS:
        for g1, g2 in combinations(order, 2):
            block = []
            for measure, sub in long[long["phase"] == phase].groupby("measure", sort=False):
                a = sub.loc[sub["genotype"] == g1, "value"].dropna()
                b = sub.loc[sub["genotype"] == g2, "value"].dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                t, pv = ttest_ind(a, b, equal_var=False)
                block.append({"phase": phase, "measure": measure, "group1": g1, "group2": g2, "mean1": a.mean(),
                              "mean2": b.mean(), "n1": len(a), "n2": len(b), "t_stat": t, "p_value": pv})
            if block:
                valid = [r for r in block if pd.notna(r["p_value"])]
                q = multipletests([r["p_value"] for r in valid], method="fdr_bh")[1] if valid else []
                for r_, qv in zip(valid, q):
                    r_["p_fdr"] = qv
                rows += block
    return pd.DataFrame(rows)


# Time-binned summary per reversal: (column, axis label)
BIN_MEASURES = [("occurrences_per_bout", "Occurrences per bout"),
                ("mean_cells", "Mean cells moved (10 = one lane)"),
                ("pct_long", "% going 10+ cells (a lane or more)"),
                ("mean_start_lane", "Mean lane where it turned around"),
                ("mean_end_lane", "Mean lane where it ended up")]


def reversals_by_bins(occurrences: pd.DataFrame, bouts: pd.DataFrame, df_hmm_bouts: pd.DataFrame, binning: str,
                      min_cells: int = 1, long_cells: int = 10) -> pd.DataFrame:
    """Per session x reversal x time bin (time_bins.BINNINGS) of each bout's
    first frame: 'occurrences_per_bout', 'pct_long' (% of occurrences
    covering >= long_cells cells), 'mean_cells' (mean cells moved per
    occurrence), 'mean_start_lane' and 'mean_end_lane' (only occurrences
    covering >= min_cells cells). `occurrences` from reversal_occurrences(),
    `bouts` from summarize_bouts(), `df_hmm_bouts` (frames with 'bout_id')
    sets the bins; bouts starting past the last frame bin are left out."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS, add_time_bins

    bin_col, _ = BINNINGS[binning]
    start = (add_time_bins(df_hmm_bouts).sort_values("s_no").groupby(["session", "bout_id"])[bin_col].first()
             .rename("time_bin").reset_index())
    b = bouts.merge(start, on=["session", "bout_id"]).dropna(subset=["time_bin"])
    o = (occurrences[occurrences["cells"] >= min_cells].merge(start, on=["session", "bout_id"])
         .dropna(subset=["time_bin"]))
    agg = o.groupby(["session", "phase", "time_bin"]).agg(
        occurrences=("cells", "size"), pct_long=("cells", lambda c: 100 * (c >= long_cells).mean()),
        mean_cells=("cells", "mean"),
        mean_start_lane=("start_lane", "mean"), mean_end_lane=("end_lane", "mean")).reset_index()
    out = b.groupby(["session", "genotype", "time_bin"]).size().rename("bouts").reset_index()
    out = out.merge(pd.DataFrame({"phase": list(REVERSALS)}), how="cross")
    out = out.merge(agg, on=["session", "phase", "time_bin"], how="left").fillna({"occurrences": 0})
    out["occurrences_per_bout"] = out["occurrences"] / out["bouts"]
    out["time_bin"] = out["time_bin"].astype(int)
    return out


def plot_reversal_bins(
    config: dict,
    binned: pd.DataFrame,
    binning: str,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Per reversal (backtrack, re-entry), the BIN_MEASURES measures by time
    bin -- a row of the first 3, then a row of the 2 lanes: one line per genotype group (mean +- SE across
    mice); a point with fewer mice than its group's n is labeled with its
    count. `binned` from reversals_by_bins()."""
    from compass_labyrinth.post_hoc_analysis.level_1.time_bins import BINNINGS, BINNING_TITLES

    _, titles = BINNINGS[binning]
    order = genotype_order(binned["genotype"])
    colors = dict(zip(order, sns.color_palette(n_colors=len(order))))
    group_n = group_sample_sizes(binned, "genotype")
    offsets = np.linspace(-0.12, 0.12, len(order)) if len(order) > 1 else [0]
    x0 = np.arange(1, len(titles) + 1)
    fig, axes = _reversal_axes((19, 18))
    for phase, title in REVERSALS.items():
        d = binned[binned["phase"] == phase]
        for ax, (col, ylabel) in zip(axes[phase], BIN_MEASURES):
            for off, g in zip(offsets, order):
                st = d[d["genotype"] == g].groupby("time_bin")[col].agg(["mean", "sem", "count"]).reindex(x0)
                ax.errorbar(x0 + off, st["mean"], yerr=st["sem"], marker="o", ms=5, capsize=3, lw=1.6,
                            color=colors[g], label=g)
                for xi, m, k in zip(x0 + off, st["mean"], st["count"]):
                    if pd.notna(m) and k < group_n.get(g, 0):
                        ax.annotate(f"n={int(k)}", (xi, m), textcoords="offset points", xytext=(5, -9),
                                    ha="left", fontsize=7, color=colors[g])
            ax.set_xticks(x0, titles, fontsize=9)
            ax.set_xlabel(f"Time in session ({BINNING_TITLES[binning]})", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(axis="y", color="0.92")
            if col.endswith("_lane"):
                ax.set_ylim(-0.3, 9.3)
                ax.set_yticks(range(10))
                ax.axhspan(6.5, 7.5, color="gold", alpha=0.15, zorder=0)
            elif col in ("mean_cells", "pct_long"):
                ax.set_ylim(0, None)
        axes[phase][0].annotate(title, xy=(0, 1.1), xycoords="axes fraction", fontsize=13, weight="bold")
    handles, labels = axes[next(iter(REVERSALS))][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=len(order), frameon=False, fontsize=10)
    append_n_to_legend(legend, group_n)
    fig.suptitle("Backtrack and re-entry over the session", fontsize=14, weight="bold", y=0.93)

    if save_fig:
        save_path = (Path(config["project_path_full"]) / figures_dirname
                     / f"reversal_distance{filename_suffix}_{binning}.pdf")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_reversals_x_time_bins": one row per session x reversal x time bin
        source_data.record(binned.assign(time_bin=binned["time_bin"].map(lambda t: titles[t - 1])))
        source_data.flush("sessions_x_reversals_x_time_bins", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def reversal_bins_tests(binned: pd.DataFrame, binning: str) -> tuple:
    """For each reversal x BIN_MEASURES measure: (mixed model measure ~
    genotype * time bin, session as random effect, with likelihood-ratio
    tests for the interaction and genotype; Welch t-tests between genotype
    groups in each bin, FDR across the bins) -- as Plot 4b's tests."""
    from compass_labyrinth.post_hoc_analysis.level_1.bout_analysis import bout_bins_mixedlm, bout_bins_ttests

    long = binned.melt(id_vars=["session", "genotype", "phase", "time_bin"], value_vars=[m for m, _ in BIN_MEASURES],
                       var_name="measure").dropna(subset=["value"])
    long["successful_bout"] = long["phase"] + ": " + long["measure"]  # the panel column those tests split by
    lmm = bout_bins_mixedlm(long, value="value").rename(columns={"bout_type": "measure"})
    tt = bout_bins_ttests(long, binning, value="value").rename(columns={"bout_type": "measure"})
    return lmm, tt


##################################################################
# Plots 6a / 6b: direction heatmaps (outbound vs. return)
##################################################################
METRICS = {
    "surveillance": {"label": "Slow / Surveillance share", "vmin": 0.3, "vmax": 1.0, "diff": 0.3, "cmap": "YlOrRd"},
    "velocity": {"label": "Belly speed", "vmin": 0, "vmax": 300, "diff": 100, "cmap": "YlGnBu"},
}


def cell_phase_table(labeled: pd.DataFrame) -> pd.DataFrame:
    """Per session x grid cell x phase: frame count, Slow / Surveillance share
    (hmm_state == 1) and mean velocity -- the per-mouse values behind the
    direction heatmaps. `labeled` from label_bout_directions() on frames with
    'hmm_state' and 'velocity'."""
    d = labeled.assign(s1=(labeled["hmm_state"] == 1).astype(float))
    return d.groupby(["session", "genotype", "grid_number", "phase"]).agg(
        n=("s1", "size"), surveillance=("s1", "mean"), velocity=("velocity", "mean")).reset_index()


# Direction maps (Plots 6a / 6b, 7 / 7b): Plot 5's phases grouped by which way the mouse travels
TRAVEL = {
    "going_in": "Going in (outbound + reward zone and overshoot moving deeper)",
    "coming_back": "Coming back (return + reward zone and overshoot moving back)",
    "backtrack": "Backtrack (backing up on the way to the reward)",
    "reentry": "Re-entry (heading back in on the path, on the way back)",
}


def travel_direction(labeled: pd.DataFrame) -> pd.Series:
    """Per frame of label_bout_directions(): going_in = outbound, overshoot
    forward and reward-zone frames moving deeper; coming_back = return,
    overshoot backward and reward-zone frames moving back toward the entry;
    backtrack; reentry; NaN for undetermined frames."""
    ph, fwd = labeled["phase"], labeled["direction"] > 0
    at = ph == "at_reward"
    t = np.select([ph.isin(["outbound", "overshoot_forward"]) | (at & fwd),
                   ph.isin(["return", "overshoot_backward"]) | (at & ~fwd),
                   ph == "backtrack", ph == "reentry"],
                  ["going_in", "coming_back", "backtrack", "reentry"], "")
    return pd.Series(t, index=labeled.index).replace("", np.nan)


def with_travel(labeled: pd.DataFrame) -> pd.DataFrame:
    """`labeled` with 'phase' replaced by travel_direction() (frames without
    one dropped) -- input for cell_phase_table() / cell_phase_angle_table()
    behind the direction maps."""
    t = travel_direction(labeled)
    return labeled.assign(phase=t)[t.notna()]


def direction_cell_values(cells: pd.DataFrame, metric: str, min_frames: int = 20) -> pd.DataFrame:
    """Per genotype x grid cell: mean across sessions of the metric for each
    TRAVEL direction, and 'change' = each session's own coming_back -
    going_in. A session counts for a cell's direction only with >= min_frames
    frames there (both, for the change). `cells` from
    cell_phase_table(with_travel(labeled))."""
    wide = cells.pivot_table(index=["session", "genotype", "grid_number"], columns="phase", values=[metric, "n"])
    per = pd.DataFrame({d: wide[(metric, d)].where(wide[("n", d)] >= min_frames)
                        for d in TRAVEL if (metric, d) in wide.columns})
    if {"going_in", "coming_back"} <= set(per.columns):
        per["change"] = per["coming_back"] - per["going_in"]
    per = per.reset_index().drop(columns="session")
    return per.groupby(["genotype", "grid_number"]).mean().reset_index()


def _maze_rows_figure(config, grid_filename, vals, order, n, rows, fmt, cbar_labels, annotate=True,
                      differences=False, no_diff_rows=()) -> plt.Figure:
    """A grid of maze heatmaps: one row per (column of `vals`, row title,
    cmap, vmin, vmax) in `rows`, one column per genotype group in `order`;
    each cell annotated with its value, grey where missing, reward outlined
    in gold, one colorbar per row. differences=True adds difference columns
    (second group minus first, difference_pairs()) to the right of each row
    except those whose column is in `no_diff_rows` (their slots are removed);
    rows sharing a value range share one difference color limit."""
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import (
        create_grid_geodata, difference_pairs, robust_diff_lim, annotate_diverging, DIFF_CMAP)
    import geopandas as gpd

    grid = create_grid_geodata(config, grid_filename)
    reward = grid[grid["FID"].isin([70, 71, 80, 81])]
    reward_outline = gpd.GeoDataFrame(geometry=[reward.union_all()], crs=grid.crs)
    pairs = difference_pairs(order) if differences else []
    n_cols = len(order) + len(pairs)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4.6 * n_cols + 1.2 * (1 + bool(pairs)), 4.35 * len(rows)),
                             squeeze=False)
    hold_draws(fig)
    by_group = {(g, col): vals[vals["genotype"] == g].set_index("grid_number")[col] for g in order for col, *_ in rows}
    diffs = {(col, b, a): by_group[(b, col)] - by_group[(a, col)]
             for col, *_ in rows if col not in no_diff_rows for b, a in pairs}
    row_range = {col: (lo, hi) for col, _, _, lo, hi in rows}
    diff_lims = {rng: robust_diff_lim([d.to_numpy() for (col, _, _), d in diffs.items() if row_range[col] == rng])
                 for rng in {row_range[col] for col, *_ in rows if col not in no_diff_rows}}
    for r, (col, row_title, cmap, lo, hi) in enumerate(rows):
        cm = plt.get_cmap(cmap)
        for c, g in enumerate(order):
            ax = axes[r, c]
            g_vals = vals[vals["genotype"] == g][["grid_number", col]]
            mapped = grid.merge(g_vals, left_on="FID", right_on="grid_number", how="left")
            mapped.plot(column=col, cmap=cmap, vmin=lo, vmax=hi, ax=ax, edgecolor="0.8", linewidth=0.6,
                        missing_kwds={"color": "0.85", "edgecolor": "0.8"})
            reward_outline.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.8)
            for _, cell in (mapped.iterrows() if annotate else []):
                v = cell[col]
                if pd.isna(v):
                    continue
                rgba = cm(min(max((v - lo) / (hi - lo), 0), 1) if hi > lo else 0.5)
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                pt = cell.geometry.centroid
                ax.annotate(fmt.format(v), (pt.x, pt.y), ha="center", va="center", fontsize=5,
                            color="white" if dark else "black")
            ax.invert_yaxis()
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{g} (n={n.get(g, 0)})", fontsize=13, weight="bold")
            if c == 0:
                ax.text(-0.06, 0.5, row_title, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=12, weight="bold")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        group_axes = list(axes[r, :len(order)])
        fig.colorbar(sm, ax=group_axes, fraction=0.025 * n_cols / len(group_axes), pad=0.02).set_label(
            cbar_labels[r], fontsize=9)
        if not pairs:
            continue
        if col in no_diff_rows:
            for ax in axes[r, len(order):]:
                fig.delaxes(ax)
            continue
        dlim = diff_lims[(lo, hi)]
        for c, (b, a) in enumerate(pairs, start=len(order)):
            ax = axes[r, c]
            mapped = grid.assign(difference=grid["FID"].map(diffs[(col, b, a)]))
            mapped.plot(column="difference", cmap=DIFF_CMAP, vmin=-dlim, vmax=dlim, ax=ax, edgecolor="0.8",
                        linewidth=0.6, missing_kwds={"color": "0.85", "edgecolor": "0.8"})
            reward_outline.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.8)
            if annotate:
                annotate_diverging(ax, mapped, "difference", dlim, fmt.replace("{:", "{:+"))
            ax.invert_yaxis()
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{b} − {a}", fontsize=13, weight="bold")
        diff_axes = list(axes[r, len(order):])
        fig.colorbar(plt.cm.ScalarMappable(cmap=DIFF_CMAP, norm=plt.Normalize(-dlim, dlim)), ax=diff_axes,
                     fraction=0.025 * n_cols / len(diff_axes), pad=0.02).set_label(
            f"{cbar_labels[r]}\ndifference (second − first)", fontsize=9)
    release_draws(fig)
    fig.maze_axes = axes[:, :len(order)]  # genotype panels, for callers drawing on top (e.g. Plot 7's arrows)
    return fig


def plot_direction_heatmaps(
    config: dict,
    cells: pd.DataFrame,
    metric: str,
    direction: str = "going_in",
    grid_filename: str = "Session0001_grid.shp",
    min_frames: int = 20,
    vmin: float | None = None,
    vmax: float | None = None,
    diff_lim: float | None = None,
    show_differences: bool = True,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Maze heatmaps of `metric` ("surveillance" or "velocity") for one
    TRAVEL direction, or "change" (coming back - going in, per mouse; two-sided
    scale, red = higher coming back): one panel per genotype group, then
    (show_differences; not for "change") group difference panels. `cells`
    from cell_phase_table(with_travel(labeled)). Cells with too few frames
    are grey; each cell is annotated with its value."""
    m = METRICS[metric]
    vmin = m["vmin"] if vmin is None else vmin
    vmax = m["vmax"] if vmax is None else vmax
    diff_lim = m["diff"] if diff_lim is None else diff_lim
    vals = direction_cell_values(cells, metric, min_frames=min_frames)
    order = genotype_order(vals["genotype"])
    n = group_sample_sizes(cells, "genotype")
    title = TRAVEL.get(direction, "Coming back \u2212 going in")
    if direction == "change":
        rows, cbar = [("change", "Coming back \u2212 going in", DIFF_CMAP, -diff_lim, diff_lim)], \
            [f"{m['label']}\n(coming back \u2212 going in)"]
    else:
        rows, cbar = [(direction, title.split(" (")[0], m["cmap"], vmin, vmax)], [m["label"]]
    if direction not in vals.columns:
        vals = vals.assign(**{direction: np.nan})
    fmt = "{:.2f}" if metric == "surveillance" else "{:.0f}"

    fig = _maze_rows_figure(config, grid_filename, vals, order, n, rows, fmt, cbar_labels=cbar,
                            differences=show_differences and direction != "change", no_diff_rows={"change"})
    fig.suptitle(f"{m['label']} -- {title} (grey = < {min_frames} frames)", fontsize=15, weight="bold")

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"direction_heatmap_{metric}_{direction}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_cells_x_phases": one row per session x grid cell x phase
        source_data.record(cells.rename(columns={"grid_number": "cell"}))
        source_data.flush("sessions_x_cells_x_phases", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def zone_direction_table(travel_cells: pd.DataFrame, zones: dict, zone_names: list) -> pd.DataFrame:
    """Per session x zone x travel direction: frames ('n') and the
    surveillance share and mean velocity over those frames (frame-weighted
    across the zone's cells). `travel_cells` from
    cell_phase_table(with_travel(labeled))."""
    parts = []
    for zone in zone_names:
        sub = travel_cells[travel_cells["grid_number"].isin(zones[zone])]
        sub = sub.assign(w_s=sub["surveillance"] * sub["n"], w_v=sub["velocity"] * sub["n"])
        g = sub.groupby(["session", "genotype", "phase"])[["n", "w_s", "w_v"]].sum().reset_index()
        g["surveillance"], g["velocity"] = g["w_s"] / g["n"], g["w_v"] / g["n"]
        parts.append(g.drop(columns=["w_s", "w_v"]).assign(zone=zone))
    return pd.concat(parts, ignore_index=True)


def plot_zone_direction_heatmap(
    config: dict,
    table: pd.DataFrame,
    zones: dict,
    zone_names: list,
    metric: str,
    direction: str,
    grid_filename: str = "Session0001_grid.shp",
    min_frames: int = 20,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """6a / 6b pooled by zone: `metric` for one TRAVEL direction in each zone
    of a zone set (per mouse, >= min_frames frames there in that direction;
    then the mean across mice), one maze panel per genotype group, then group
    difference panels (second minus first, per zone); cells outside the set
    hatched. `table` from zone_direction_table()."""
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import (
        create_grid_geodata, genotype_panel_axes, robust_diff_lim, DIFF_CMAP)
    import geopandas as gpd

    m = METRICS[metric]
    t = table[(table["phase"] == direction) & (table["n"] >= min_frames)]
    vals = t.groupby(["genotype", "zone"])[metric].mean()
    order = genotype_order(table["genotype"])
    n = group_sample_sizes(table, "genotype")
    grid = create_grid_geodata(config, grid_filename)
    in_set = set().union(*(zones[z] for z in zone_names))
    reward = gpd.GeoDataFrame(geometry=[grid[grid["FID"].isin([70, 71, 80, 81])].union_all()], crs=grid.crs)
    fmt = "{:.2f}" if metric == "surveillance" else "{:.0f}"
    zone_vals = {g: {z: vals.get((g, z), np.nan) for z in zone_names} for g in order}

    fig, panels = genotype_panel_axes(order, one_row=True, differences=True, panel_w=4.6, panel_h=4.6)
    diffs = {(sp[1], sp[2]): {z: zone_vals[sp[1]][z] - zone_vals[sp[2]][z] for z in zone_names}
             for _, sp in panels if sp[0] == "diff"}
    dlim = robust_diff_lim([np.array(list(d.values())) for d in diffs.values()])
    for ax, spec in panels:
        if spec[0] == "group":
            values, cmap, norm, f = zone_vals[spec[1]], m["cmap"], plt.Normalize(m["vmin"], m["vmax"]), fmt
            ax.set_title(f"{spec[1]} (n={n.get(spec[1], 0)})", fontsize=13, weight="bold")
            cb_label = m["label"]
        else:
            values, cmap, norm = diffs[(spec[1], spec[2])], DIFF_CMAP, plt.Normalize(-dlim, dlim)
            f = "{:+.2f}" if metric == "surveillance" else "{:+.0f}"
            ax.set_title(f"{spec[1]} \u2212 {spec[2]}", fontsize=13, weight="bold")
            cb_label = f"{m['label']}\n{spec[1]} \u2212 {spec[2]}"
        cm = plt.get_cmap(cmap)
        grid[~grid["FID"].isin(in_set)].plot(ax=ax, facecolor="0.92", edgecolor="0.8", hatch="///", linewidth=0.4)
        for zone in zone_names:
            shape = grid[grid["FID"].isin(zones[zone])]
            v = values[zone]
            shape.plot(ax=ax, color="0.8" if pd.isna(v) else cm(norm(v)), edgecolor="0.85", linewidth=0.4)
            if pd.notna(v):
                rgba = cm(norm(v))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                for _, cell in shape.iterrows():
                    pt = cell.geometry.centroid
                    ax.annotate(f.format(v), (pt.x, pt.y), ha="center", va="center", fontsize=5.5,
                                color="white" if dark else "black")
        reward.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.8)
        ax.invert_yaxis()
        ax.axis("off")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax).set_label(cb_label, fontsize=8)
    release_draws(fig)
    fig.suptitle(f"{m['label']} -- {TRAVEL[direction]} by zone (grey = < {min_frames} frames; hatched = not in zone set)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"direction_heatmap_{metric}_{direction}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_zones_x_directions": one row per session x zone x direction
        source_data.record(table[table["phase"] == direction])
        source_data.flush("sessions_x_zones_x_directions", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def zone_direction_ttests(table: pd.DataFrame, metric: str, direction: str, min_frames: int = 20) -> pd.DataFrame:
    """For one TRAVEL direction: Welch t-tests between every pair of genotype
    groups per zone (each mouse's value there, >= min_frames frames),
    Benjamini-Hochberg FDR across the zones per pair."""
    from itertools import combinations
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    t = table[(table["phase"] == direction) & (table["n"] >= min_frames)]
    order = genotype_order(table["genotype"])
    rows = []
    for g1, g2 in combinations(order, 2):
        block = []
        for zone, sub in t.groupby("zone", sort=False):
            a = sub.loc[sub["genotype"] == g1, metric].dropna()
            b = sub.loc[sub["genotype"] == g2, metric].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            tt, pv = ttest_ind(a, b, equal_var=False)
            block.append({"direction": direction, "zone": zone, "group1": g1, "group2": g2, "mean1": a.mean(),
                          "mean2": b.mean(), "n1": len(a), "n2": len(b), "t_stat": tt, "p_value": pv})
        if block:
            for r, q in zip(block, multipletests([r["p_value"] for r in block], method="fdr_bh")[1]):
                r["p_fdr"] = q
            rows += block
    return pd.DataFrame(rows)


##################################################################
# Plot 5f: where mice turn around (per pass through a cell)
##################################################################
# Plot 5f turns: (phases a pass can turn from, phases it turns into)
TURN_BACK = ({"outbound", "overshoot_forward", "at_reward"}, {"backtrack", "overshoot_backward", "at_reward"})
TURN_IN = ({"return", "overshoot_backward", "at_reward"}, {"reentry", "overshoot_forward", "at_reward"})


def cell_turn_table(labeled: pd.DataFrame) -> pd.DataFrame:
    """Per session x grid cell: how often a pass through the cell turns
    around there. A pass = one stay in the cell; it enters with the direction
    and phase of the frame before it. Turn back (forward -> backward):
    passes entering while going deeper (outbound, overshoot forward, reward
    zone) -- 'back_entries' -- and those with a switch to backtrack /
    overshoot backward / reward zone moving back in the cell -- 'back_turns'.
    Turn in again (backward -> forward): passes entering while coming back
    (return, overshoot backward, reward zone) -- 'in_entries' -- and those
    switching to re-entry / overshoot forward / reward zone moving deeper --
    'in_turns'. So the furthest point of a bout that never reaches the reward
    (its switch is to return) is not a turn back. `labeled` from
    label_bout_directions()."""
    d = labeled.sort_values(["session", "s_no"]).reset_index(drop=True)
    sign = np.sign(d["direction"]).to_numpy()
    ph = d["phase"].to_numpy()
    same_bout = ((d["session"] == d["session"].shift()) & (d["bout_id"] == d["bout_id"].shift())).to_numpy()
    visit = np.cumsum(~same_bout | (d["grid_number"] != d["grid_number"].shift()).to_numpy())
    prev_sign = np.where(same_bout, np.roll(sign, 1), sign)
    prev_phase = np.where(same_bout, np.roll(ph, 1), ph)
    v = pd.DataFrame({
        "visit": visit, "session": d["session"], "genotype": d["genotype"], "grid_number": d["grid_number"],
        "entry_sign": prev_sign, "entry_phase": prev_phase,
        "turn_back": (prev_sign > 0) & (sign < 0) & np.isin(ph, list(TURN_BACK[1])),
        "turn_in": (prev_sign < 0) & (sign > 0) & np.isin(ph, list(TURN_IN[1])),
    }).groupby("visit").agg(session=("session", "first"), genotype=("genotype", "first"),
                            grid_number=("grid_number", "first"), entry_sign=("entry_sign", "first"),
                            entry_phase=("entry_phase", "first"), turn_back=("turn_back", "any"),
                            turn_in=("turn_in", "any"))
    back = (v["entry_sign"] > 0) & v["entry_phase"].isin(TURN_BACK[0])
    inn = (v["entry_sign"] < 0) & v["entry_phase"].isin(TURN_IN[0])
    v = v.assign(back_entries=back, back_turns=back & v["turn_back"], in_entries=inn, in_turns=inn & v["turn_in"])
    cols = ["back_entries", "back_turns", "in_entries", "in_turns"]
    return v.groupby(["session", "genotype", "grid_number"])[cols].sum().reset_index()


def reversal_cell_values(turns: pd.DataFrame, min_passes: int = 5) -> pd.DataFrame:
    """Per genotype x grid cell, mean across sessions of 'p_backtrack' (share
    of passes going deeper that turned back in the cell) and 'p_reentry'
    (share of passes coming back that turned in again). A session counts only
    with >= min_passes such passes. `turns` from cell_turn_table()."""
    per = pd.DataFrame({
        "genotype": turns["genotype"], "grid_number": turns["grid_number"],
        "p_backtrack": (turns["back_turns"] / turns["back_entries"]).where(turns["back_entries"] >= min_passes),
        "p_reentry": (turns["in_turns"] / turns["in_entries"]).where(turns["in_entries"] >= min_passes),
    })
    return per.groupby(["genotype", "grid_number"])[["p_backtrack", "p_reentry"]].mean().reset_index()


def plot_reversal_heatmaps(
    config: dict,
    turns: pd.DataFrame,
    grid_filename: str = "Session0001_grid.shp",
    min_passes: int = 5,
    vmax: float = 0.2,
    show_differences: bool = True,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """Maze heatmaps of how often a pass through each cell turns around
    there (cell_turn_table()): row 1 = turn back (forward -> backward: a
    backtrack on the path, or turning back in the reward zone / overshoot),
    row 2 = turn in again (backward -> forward: a re-entry on the path, or
    in the reward zone / overshoot); one column per genotype group, then
    (show_differences) group difference panels. Both rows on one 0-vmax scale."""
    vals = reversal_cell_values(turns, min_passes=min_passes)
    order = genotype_order(vals["genotype"])
    n = group_sample_sizes(turns, "genotype")
    rows = [("p_backtrack", "Backtrack (turn back)", "Oranges", 0, vmax),
            ("p_reentry", "Re-entry (turn in again)", "Oranges", 0, vmax)]  # same scale: rows comparable
    fig = _maze_rows_figure(config, grid_filename, vals, order, n, rows, "{:.2f}",
                            cbar_labels=["P(turn back)\nper pass going deeper", "P(turn in again)\nper pass coming back"],
                            differences=show_differences)
    fig.suptitle(f"How often a pass through the cell turns around (grey = < {min_passes} passes)",
                 fontsize=15, weight="bold")

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"reversal_heatmap{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_cells_turns": per session x cell, passes and turns each way
        source_data.record(turns.rename(columns={"grid_number": "cell"}))
        source_data.flush("sessions_x_cells_turns", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


##################################################################
# Plot 7: signed head angle by direction of travel, per zone
##################################################################
PHASE_TITLES = {**TRAVEL, "outbound": "Outbound (going in)", "backtrack": "Backtrack (backing up on the way in)",
                "overshoot_forward": "Overshoot forward (going deeper past the reward)",
                "overshoot_backward": "Overshoot backward (coming back toward the reward)",
                "return": "Return (coming back)", "reentry": "Re-entry (heading in again on the way out)"}


def zone_phase_angle_table(labeled: pd.DataFrame, kinematics: pd.DataFrame, zones: dict,
                           zone_names: list) -> pd.DataFrame:
    """Per session x zone x phase: number of frames with a head angle and their
    mean signed head angle (+ = the mouse's right). `labeled` from
    label_bout_directions(), `kinematics` from add_nose_speeds() (matched on
    session + s_no). A cell in several zones counts toward each."""
    d = labeled[["session", "s_no", "genotype", "grid_number", "phase"]].merge(
        kinematics[["session", "s_no", "head_angle"]], on=["session", "s_no"], how="left")
    d = d.dropna(subset=["head_angle"])
    parts = []
    for zone in zone_names:
        sub = d[d["grid_number"].isin(zones[zone])]
        g = sub.groupby(["session", "genotype", "phase"]).agg(n=("head_angle", "size"),
                                                             head_angle=("head_angle", "mean")).reset_index()
        parts.append(g.assign(zone=zone))
    return pd.concat(parts, ignore_index=True)


def zone_angle_ttests(table: pd.DataFrame, phase: str, min_frames: int = 20) -> pd.DataFrame:
    """For one phase: Welch t-tests between every pair of genotype groups per
    zone (each mouse's mean signed head angle there), Benjamini-Hochberg FDR
    across the zones of the set."""
    from itertools import combinations
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    t = table[(table["phase"] == phase) & (table["n"] >= min_frames)]
    order = genotype_order(t["genotype"])
    rows = []
    for g1, g2 in combinations(order, 2):
        block = []
        for zone, z in t.groupby("zone", sort=False):
            a = z.loc[z["genotype"] == g1, "head_angle"]
            b = z.loc[z["genotype"] == g2, "head_angle"]
            if len(a) < 2 or len(b) < 2:
                continue
            stat, pval = ttest_ind(a, b, equal_var=False)
            block.append({"phase": phase, "zone": zone, "group1": g1, "group2": g2, "mean1": a.mean(),
                          "mean2": b.mean(), "n1": len(a), "n2": len(b), "t_stat": stat, "p_value": pval})
        if block:
            for r, q in zip(block, multipletests([r["p_value"] for r in block], method="fdr_bh")[1]):
                r["p_fdr"] = q
            rows += block
    return pd.DataFrame(rows)


def plot_zone_head_angle(
    config: dict,
    table: pd.DataFrame,
    zones: dict,
    zone_names: list,
    phase: str,
    grid_filename: str = "Session0001_grid.shp",
    min_frames: int = 20,
    lim: float = 30,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """One phase's mean signed head angle per zone (+ = right, red; - = left,
    blue), one maze panel per genotype group; cells outside the zone set are
    hatched grey. Each mouse needs >= min_frames frames in a zone for that phase."""
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import create_grid_geodata
    import geopandas as gpd

    t = table[(table["phase"] == phase) & (table["n"] >= min_frames)]
    vals = t.groupby(["genotype", "zone"])["head_angle"].mean()
    order = genotype_order(table["genotype"])
    n = group_sample_sizes(table, "genotype")
    grid = create_grid_geodata(config, grid_filename)
    in_set = set().union(*(zones[z] for z in zone_names))
    reward = gpd.GeoDataFrame(geometry=[grid[grid["FID"].isin([70, 71, 80, 81])].union_all()], crs=grid.crs)
    cmap, norm = plt.get_cmap("RdBu_r"), plt.Normalize(-lim, lim)

    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order) + 1.2, 4.6), squeeze=False)
    for ax, g in zip(axes[0], order):
        grid[~grid["FID"].isin(in_set)].plot(ax=ax, facecolor="0.92", edgecolor="0.8", hatch="///", linewidth=0.4)
        for zone in zone_names:
            shape = grid[grid["FID"].isin(zones[zone])]
            v = vals.get((g, zone), np.nan)
            color = "0.8" if pd.isna(v) else cmap(norm(v))
            shape.plot(ax=ax, color=color, edgecolor="0.85", linewidth=0.4)
            outline = shape.union_all()
            gpd.GeoSeries([outline], crs=grid.crs).boundary.plot(ax=ax, color="0.3", linewidth=0.8)
            if pd.notna(v):
                parts = list(outline.geoms) if hasattr(outline, "geoms") else [outline]
                rgba = cmap(norm(v))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                for part in parts:
                    pt = part.representative_point()
                    ax.annotate(f"{v:.0f}", (pt.x, pt.y), ha="center", va="center", fontsize=6,
                                color="white" if dark else "black")
        reward.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.8)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{g} (n={n.get(g, 0)})", fontsize=13, weight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=list(axes[0]), fraction=0.025, pad=0.02).set_label(
        "Signed head angle (deg)\n+ = right, - = left", fontsize=9)
    fig.suptitle(f"Head angle -- {PHASE_TITLES[phase]} (grey = < {min_frames} frames; hatched = not in zone set)",
                 fontsize=13, weight="bold")

    if save_fig:
        save_path = Path(config["project_path_full"]) / figures_dirname / f"head_angle_{phase}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_zones_x_phases": one row per session x zone x phase
        source_data.record(table[table["phase"] == phase])
        source_data.flush("sessions_x_zones_x_phases", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def plot_head_angle_cells(
    config: dict,
    angles: pd.DataFrame,
    phase: str,
    grid_filename: str = "Session0001_grid.shp",
    min_frames: int = 20,
    lim: float = 30,
    absolute: bool = False,
    show_differences: bool = True,
    diff_lim: float | None = None,
    filename_suffix: str = "",
    figures_dirname: str = "figures",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """One phase's mean signed head angle per grid cell (+ = the mouse's
    right, red; - = left, blue), one maze panel per genotype group, with a
    body arrow and head arrow per cell. absolute=True (Plot 7b) colors and
    labels the cells by the absolute value of that same signed mean (how
    strongly the head leans, ignoring the side; 0-lim); the arrows still use
    the signed mean.
    `angles` from cell_phase_angle_table(); a mouse counts for a cell only with
    >= min_frames frames there in that phase, else the cell is grey.
    show_differences adds difference panels (second group minus first; for
    7b, |second| - |first|) in genotype_panel_axes()'s one-row layout; their
    color limit is diff_lim or, if None, the 95th percentile of |difference|."""
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import (
        create_grid_geodata, genotype_panel_axes, robust_diff_lim, annotate_diverging, DIFF_CMAP)
    import geopandas as gpd

    t = angles[(angles["phase"] == phase) & (angles["n"] >= min_frames)]
    means = t.groupby(["genotype", "grid_number"])["head_angle"].mean()
    shown = means.abs() if absolute else means
    order = genotype_order(angles["genotype"])
    n = group_sample_sizes(angles, "genotype")
    cmap, lo, hi = ("magma_r", 0, lim) if absolute else ("RdBu_r", -lim, lim)
    label = "|Signed head angle| (deg)\nlean, either side" if absolute else "Signed head angle (deg)\n+ = right, - = left"
    grid = create_grid_geodata(config, grid_filename)
    reward = gpd.GeoDataFrame(geometry=[grid[grid["FID"].isin([70, 71, 80, 81])].union_all()], crs=grid.crs)

    fig, panels = genotype_panel_axes(order, one_row=True, differences=show_differences, panel_w=4.8, panel_h=4.6)
    cell_vals = {g: shown.xs(g, level="genotype") if g in shown.index.get_level_values(0) else pd.Series(dtype=float)
                 for g in order}
    diffs = {(sp[1], sp[2]): cell_vals[sp[1]] - cell_vals[sp[2]] for _, sp in panels if sp[0] == "diff"}
    dlim = diff_lim if diff_lim is not None else robust_diff_lim([d.to_numpy() for d in diffs.values()])
    group_axes = []
    for ax, spec in panels:
        if spec[0] == "group":
            g = spec[1]
            mapped = grid.assign(value=grid["FID"].map(cell_vals[g]))
            mapped.plot(column="value", cmap=cmap, vmin=lo, vmax=hi, ax=ax, edgecolor="0.8", linewidth=0.6,
                        missing_kwds={"color": "0.85", "edgecolor": "0.8"})
            ax.set_title(f"{g} (n={n.get(g, 0)})", fontsize=13, weight="bold")
            sm, cb_label = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)), label
            group_axes.append((ax, g))
        else:
            b, a = spec[1], spec[2]
            mapped = grid.assign(difference=grid["FID"].map(diffs[(b, a)]))
            mapped.plot(column="difference", cmap=DIFF_CMAP, vmin=-dlim, vmax=dlim, ax=ax, edgecolor="0.8",
                        linewidth=0.6, missing_kwds={"color": "0.85", "edgecolor": "0.8"})
            annotate_diverging(ax, mapped, "difference", dlim, "{:+.0f}")
            ax.set_title(f"{b} \u2212 {a}", fontsize=13, weight="bold")
            sm = plt.cm.ScalarMappable(cmap=DIFF_CMAP, norm=plt.Normalize(-dlim, dlim))
            cb_label = ("|angle| difference (deg)" if absolute else "angle difference (deg)") + f"\n{b} \u2212 {a}"
        reward.plot(ax=ax, facecolor="none", edgecolor="gold", linewidth=1.8)
        ax.invert_yaxis()
        ax.axis("off")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(sm, cax=cax).set_label(cb_label, fontsize=8)
    _draw_head_arrows(group_axes, config, grid_filename, t, phase, absolute, cmap, lo, hi)
    release_draws(fig)
    title = "|Head angle|" if absolute else "Head angle"
    fig.suptitle(f"{title} -- {PHASE_TITLES[phase]} (grey = < {min_frames} frames)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        stem = "head_angle_abs" if absolute else "head_angle"
        save_path = Path(config["project_path_full"]) / figures_dirname / f"{stem}_{phase}{filename_suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # "sessions_x_cells_x_phases": shared with 6a-6c (head_angle column)
        source_data.record(angles[angles["phase"] == phase].rename(columns={"grid_number": "cell"}))
        source_data.flush("sessions_x_cells_x_phases", figure=save_path.stem)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()
    elif not return_fig:
        plt.close(fig)  # not shown -> don't let Jupyter display it at cell end

    if return_fig:
        return fig


def cell_phase_angle_table(labeled: pd.DataFrame, kinematics: pd.DataFrame) -> pd.DataFrame:
    """Per session x grid cell x phase: number of frames with a head angle
    and their mean signed head angle. `labeled` from label_bout_directions(),
    `kinematics` from add_nose_speeds() (matched on session + s_no -- the two
    tables' row orders differ)."""
    d = labeled[["session", "s_no", "genotype", "grid_number", "phase"]].merge(
        kinematics[["session", "s_no", "head_angle", "head_angle_abs", "nose_x", "nose_y", "sternum_x",
                    "sternum_y", "tailbase_x", "tailbase_y"]],
        on=["session", "s_no"], how="left")
    d = d.dropna(subset=["head_angle"])
    # Head direction in the maze: unit vector sternum -> nose (video coordinates)
    vx, vy = d["nose_x"] - d["sternum_x"], d["nose_y"] - d["sternum_y"]
    norm = np.hypot(vx, vy).replace(0, np.nan)
    # Body axis in the maze: unit vector tail base -> sternum
    bx, by = d["sternum_x"] - d["tailbase_x"], d["sternum_y"] - d["tailbase_y"]
    bnorm = np.hypot(bx, by).replace(0, np.nan)
    d = d.assign(head_dx=vx / norm, head_dy=vy / norm, body_dx=bx / bnorm, body_dy=by / bnorm)
    return d.groupby(["session", "genotype", "grid_number", "phase"]).agg(
        n=("head_angle", "size"), head_angle=("head_angle", "mean"), head_angle_abs=("head_angle_abs", "mean"),
        head_dx=("head_dx", "mean"), head_dy=("head_dy", "mean"),
        body_dx=("body_dx", "mean"), body_dy=("body_dy", "mean")).reset_index()


FORWARD_PHASES = {"outbound", "reentry", "overshoot_forward", "going_in"}  # moving deeper; the others move toward the entry


def _stage_directions(grid) -> dict:
    """{cell: (dx, dy)} unit direction of forward travel along the maze path
    in each cell, in grid coordinates: from the previous cell to the next one
    in FORWARD_PATH (one-sided at the two ends), so straight lanes and U-turns
    both get the way the path actually runs through them."""
    cen = {c: (geom.centroid.x, geom.centroid.y) for c, geom in grid.geometry.items()}
    out = {}
    for i, cell in enumerate(FORWARD_PATH):
        a = FORWARD_PATH[max(i - 1, 0)]
        b = FORWARD_PATH[min(i + 1, len(FORWARD_PATH) - 1)]
        if a not in cen or b not in cen:
            continue
        dx, dy = cen[b][0] - cen[a][0], cen[b][1] - cen[a][1]
        length = np.hypot(dx, dy)
        if length > 0:
            out[cell] = (dx / length, dy / length)
    return out


def _draw_head_arrows(group_axes, config, grid_filename, t, phase, absolute, cmap, lo, hi) -> None:
    """On each Plot 7 / 7b panel, per cell: a grey line from behind the center
    up to it along the stage's direction of travel along the maze path (toward
    the reward for outbound / re-entry, toward the entry for backtrack /
    return), a dark arrow continuing from the center along that direction
    turned by the cell's mean SIGNED head angle (+ = right), and that angle --
    or its absolute value if `absolute` -- in the bottom-right corner (colored
    by `cmap` lo..hi)."""
    from compass_labyrinth.post_hoc_analysis.level_1.grid_heatmap import create_grid_geodata

    grid = create_grid_geodata(config, grid_filename).set_index("FID")
    travel = _stage_directions(grid)
    sign = 1 if phase in FORWARD_PHASES else -1
    cm, norm = plt.get_cmap(cmap), plt.Normalize(lo, hi)
    for ax, g in group_axes:
        sub = t[t["genotype"] == g].groupby("grid_number")["head_angle"].mean()
        for cell, angle in sub.items():
            value = abs(angle) if absolute else angle
            if cell not in grid.index or cell not in travel or pd.isna(angle):
                continue
            minx, miny, maxx, maxy = grid.loc[cell].geometry.bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            half = 0.4 * min(maxx - minx, maxy - miny)
            tx, ty = travel[cell][0] * sign, travel[cell][1] * sign
            # Turn by the head angle; the axes show y downward (as in the
            # video), where this rotation turns toward the mouse's right
            th = np.radians(angle)
            hx, hy = tx * np.cos(th) - ty * np.sin(th), tx * np.sin(th) + ty * np.cos(th)
            rgba = cm(norm(value))
            dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
            ink = "white" if dark else "0.15"
            # Body arrow (tail base -> sternum): ends at the center, the sternum, where the head arrow starts
            ax.annotate("", xy=(cx, cy), xytext=(cx - tx * half, cy - ty * half),
                        arrowprops=dict(arrowstyle="-|>", color="0.85" if dark else "0.55", lw=0.9,
                                        mutation_scale=5, shrinkA=0, shrinkB=0), zorder=3)
            # Head arrow from the center
            ax.annotate("", xy=(cx + hx * half, cy + hy * half), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.0, mutation_scale=6, shrinkA=0, shrinkB=0),
                        zorder=4)
            # bottom-right as displayed: the y axis is inverted, so larger y is lower
            pad = 0.06 * (maxx - minx)
            ax.text(maxx - pad, maxy - pad, f"{value:.0f}", ha="right", va="bottom", fontsize=3.6, color=ink)