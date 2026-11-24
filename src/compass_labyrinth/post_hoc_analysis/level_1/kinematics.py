"""
NOSE KINEMATICS (allocentric and egocentric speed)
Goal:
    ├── Nose, sternum and tail base positions for every analyzed frame, from the
    │     DLC output the pipeline was built from (SessionXXXX_withGrids.csv).
    ├── Allocentric nose speed: how fast the nose moves relative to the maze.
    └── Egocentric nose speed: how fast the nose moves relative to the body
          (origin = sternum, forward axis = tail base -> sternum) -- head movement
          with the body's own translation and rotation removed. The sternum,
          not the belly, gives the longer, steadier body axis (~96 vs ~44 px)
          and sits closer to the nose, so tracking wobble tilts the frame less.

Same rules as the pipeline's velocity column: a point is used only on frames
where its likelihood is > 0.6 (frames below are dropped for that measure, not
interpolated), and speed = distance between consecutive kept rows x fps.
"""

from pathlib import Path

import numpy as np
import pandas as pd

KEYPOINTS = ("Nose", "Sternum", "Tailbase")


def load_keypoints(config: dict, session: int, keypoints=KEYPOINTS) -> pd.DataFrame:
    """One session's x, y and likelihood for `keypoints`, one row per video
    frame, with 's_no' = 1-based row number -- the same numbering as the
    pipeline's s_no, so the rows join straight onto data_with_states.csv."""
    folder = Path(config.get("dlc_results_path") or Path(config["project_path_full"]) / "data" / "dlc_results")
    path = folder / f"Session{int(session):04d}_withGrids.csv"
    raw = pd.read_csv(path, header=[0, 1, 2], index_col=0, low_memory=False)
    raw.columns = raw.columns.droplevel(0)
    out = pd.DataFrame({"s_no": np.arange(1, len(raw) + 1)})
    for kp in keypoints:
        for coord in ("x", "y", "likelihood"):
            out[f"{kp.lower()}_{coord}"] = pd.to_numeric(raw[(kp, coord)], errors="coerce").to_numpy()
    return out


def _consecutive_speed(points: np.ndarray, keep: np.ndarray, fps: float) -> np.ndarray:
    """Distance between consecutive kept rows x fps (the pipeline's velocity
    rule), NaN on rows that aren't kept and on each kept run's first row."""
    speed = np.full(len(points), np.nan)
    idx = np.flatnonzero(keep)
    if len(idx) > 1:
        step = np.hypot(*(points[idx[1:]] - points[idx[:-1]]).T) * fps
        speed[idx[1:]] = step
    return speed


def add_nose_speeds(
    df: pd.DataFrame,
    config: dict,
    likelihood_threshold: float = 0.6,
    fps: float = 5.0,
) -> pd.DataFrame:
    """`df` (analyzed frames with 'session' and 's_no', e.g. df_hmm) plus the
    three keypoints' x/y/likelihood and:
        nose_speed_allo -- nose speed relative to the maze (tracking units/s)
        sternum_speed   -- sternum speed relative to the maze, for comparison
                           with the belly's 'velocity'
        nose_speed_ego  -- nose speed in the body frame: the nose's position
                           relative to the sternum, rotated so the forward axis
                           points from the tail base to the sternum
    Each speed uses only rows where its points are all above the likelihood
    threshold (nose for allo; nose, sternum and tail base for ego); within a
    session, distance between consecutive such rows x fps."""
    parts = []
    for session, rows in df.groupby("session", sort=False):
        kp = load_keypoints(config, session)
        merged = rows.merge(kp, on="s_no", how="left")
        merged.index = rows.index
        nose = merged[["nose_x", "nose_y"]].to_numpy(float)
        nose_ok = (merged["nose_likelihood"] > likelihood_threshold).to_numpy()
        merged["nose_speed_allo"] = _consecutive_speed(nose, nose_ok, fps)
        merged["sternum_speed"] = _consecutive_speed(
            merged[["sternum_x", "sternum_y"]].to_numpy(float),
            (merged["sternum_likelihood"] > likelihood_threshold).to_numpy(), fps)

        sternum = merged[["sternum_x", "sternum_y"]].to_numpy(float)
        tail = merged[["tailbase_x", "tailbase_y"]].to_numpy(float)
        heading = sternum - tail
        norm = np.hypot(*heading.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd = heading / norm[:, None]
        left = np.column_stack([-fwd[:, 1], fwd[:, 0]])
        rel = nose - sternum
        ego = np.column_stack([(rel * fwd).sum(axis=1), (rel * left).sum(axis=1)])
        all_ok = (nose_ok
                  & (merged["sternum_likelihood"] > likelihood_threshold).to_numpy()
                  & (merged["tailbase_likelihood"] > likelihood_threshold).to_numpy()
                  & (norm > 0))
        merged["nose_ego_forward"] = np.where(all_ok, ego[:, 0], np.nan)
        merged["nose_ego_lateral"] = np.where(all_ok, ego[:, 1], np.nan)
        merged["nose_speed_ego"] = _consecutive_speed(ego, all_ok, fps)
        # Head angle: direction of the nose from the sternum relative to the
        # body axis (tail base -> sternum), degrees; 0 = straight ahead.
        # Image coordinates (y down): positive = the mouse's right, assuming
        # an unmirrored top-down camera view.
        merged["head_angle"] = np.degrees(np.arctan2(merged["nose_ego_lateral"], merged["nose_ego_forward"]))
        merged["head_angle_abs"] = merged["head_angle"].abs()
        parts.append(merged)
    return pd.concat(parts).loc[df.index]


def compute_mean_metric(df: pd.DataFrame, genotype_name: str, column: str) -> pd.DataFrame:
    """Mean of `column` per grid cell for one genotype (rows where it's NaN
    skipped), in compute_mean_velocity()'s output shape with the value under
    `column` -- a drop-in compute_fn for plot_all_genotype_heatmaps()
    (pass value_col=column)."""
    d = df[(df["genotype"] == genotype_name) & df[column].notna()]
    out = d.groupby(["genotype", "grid_number"]).agg(**{column: (column, "mean"), "x": ("x", "mean"), "y": ("y", "mean")})
    return out.reset_index()
