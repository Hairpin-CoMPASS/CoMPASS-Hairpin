"""
Align each session's grid columns to the maze walls seen in its video.

The grid from ``batch_create_grids`` spans the DLC crop box, so a crop that is
shifted or wider than the maze puts the grid lines off the walls. This module
finds the inner walls in a background image of the video (median of frames
spread over the session), fits evenly spaced columns ``x0 + k * cell`` to them,
and rewrites each frame's column in ``<session>_withGrids.csv``. Rows are kept.

Outputs go to ``<project>/grid_alignment/``:

- ``alignment.csv``: one row per session (fit, status, frames re-assigned)
- ``<session>_alignment.jpg``: one video frame with the fitted grid (green,
  cell numbers), the crop grid (red), entry/target cells and the tracked point
- ``withGrids_original/``: untouched copies of the crop-grid ``_withGrids.csv``
  (re-alignment always starts from these)

Sessions already in ``alignment.csv`` with status ``aligned`` are skipped.
"""

from __future__ import annotations

import csv
import glob
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ALIGN_DIRNAME = "grid_alignment"
VIDEO_COLUMNS = ("video path", "video_path", "video")


def _sha1(path):
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_background(video_path, crop, n_frames=9, span=(0.2, 0.9)):
    """Median of ``n_frames`` cropped frames spread over ``span`` of the video (the mouse averages out)."""
    import cv2

    x1, x2, y1, y2 = crop
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fi in np.linspace(n * span[0], n * span[1], n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if ok:
            frames.append(f[y1:y2, x1:x2])
    cap.release()
    if not frames:
        raise IOError(f"could not read frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def fit_grid_to_walls(background, n_cols=10, band=(0.25, 0.6), prominence=8, n_iter=3):
    """
    Fit the inner walls (bright vertical lines) of a background image to ``x0 + k * cell``, k = 1..n_cols-1.

    Returns dict(x0, x_right, cell_px, n_walls, max_resid_px), in crop pixel coordinates.
    """
    import cv2
    from scipy.signal import find_peaks

    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32) if background.ndim == 3 else background.astype(np.float32)
    H, W = gray.shape
    prof = gray[int(H * band[0]):int(H * band[1])].mean(axis=0)
    hp = prof - pd.Series(prof).rolling(31, center=True, min_periods=1).median().to_numpy()
    peaks, _ = find_peaks(hp, distance=W / (2 * n_cols), prominence=prominence)
    x0, cell = 0.0, W / n_cols  # start from the crop grid
    ks, xs = [], []
    for _ in range(n_iter):
        ks, xs = [], []
        for k in range(1, n_cols):
            near = peaks[np.abs(peaks - (x0 + k * cell)) < cell / 2.5]
            if len(near):
                ks.append(k)
                xs.append(near[np.argmin(np.abs(near - (x0 + k * cell)))])
        if len(ks) < 2:
            break
        cell, x0 = np.polyfit(ks, xs, 1)
    resid = float(np.abs(np.array(xs) - (x0 + cell * np.array(ks))).max()) if len(ks) >= 2 else np.inf
    return dict(x0=float(x0), x_right=float(x0 + n_cols * cell), cell_px=float(cell), n_walls=len(ks), max_resid_px=resid)


def _crop_for_session(dlc_dir, session, scorer, meta_row):
    """The crop DLC actually used (meta.pickle), else the metadata X1/X2/Y1/Y2."""
    pk = glob.glob(str(Path(dlc_dir) / f"{session}{scorer}_meta.pickle")) or glob.glob(str(Path(dlc_dir) / f"{session}DLC_*_meta.pickle"))
    if pk:
        crop = pickle.load(open(pk[0], "rb"))["data"].get("cropping_parameters")
        if crop:
            return tuple(int(v) for v in crop)
    return tuple(int(meta_row[c]) for c in ("x1", "x2", "y1", "y2"))


def _video_for_session(meta_row, project_path):
    for c in VIDEO_COLUMNS:
        if c in meta_row and isinstance(meta_row[c], str) and meta_row[c].strip():
            p = Path(meta_row[c].strip())
            return p if p.is_absolute() else Path(project_path).parent / p
    return None


def _reassign_columns(src, dst, fit, n_cols, rotate_180):
    """Rewrite every bodypart's grid_number column in a DLC ``_withGrids.csv``; rows (cell % n_cols) are kept.
    Returns (frames, changed vs src, changed vs the current dst)."""
    lines = list(csv.reader(open(src, newline="")))
    current = list(csv.reader(open(dst, newline=""))) if Path(dst).exists() else lines
    bp, co = lines[1], lines[2]
    pairs = [(j, next(i for i in range(len(bp)) if bp[i] == bp[j] and co[i] == "x")) for j in range(len(bp)) if co[j] == "grid_number"]
    n = ch_src = ch_cur = 0
    for r, row in enumerate(lines[3:], start=3):
        for jg, jx in pairs:
            if row[jg] == "" or row[jx] == "":
                continue
            old = int(float(row[jg]))
            x = float(row[jx])
            col = (fit["x_right"] - x) / fit["cell_px"] if rotate_180 else (x - fit["x0"]) / fit["cell_px"]
            new = min(n_cols - 1, max(0, math.floor(col))) * n_cols + old % n_cols
            row[jg] = str(new)
            n += 1
            ch_src += new != old
            if r < len(current) and current[r][jg] != "" and int(float(current[r][jg])) != new:
                ch_cur += 1
    with open(dst, "w", newline="") as fh:
        csv.writer(fh).writerows(lines)
    return n, ch_src, ch_cur


def _draw_overlay(video_path, crop, withgrids, fit, n_cols, rotate_180, entry, target, out_path, label):
    """One confident mid-session frame: fitted grid (green, numbered), crop grid (red), entry/target cells, tracked point."""
    import cv2

    x1, x2, y1, y2 = crop
    W, H = x2 - x1, y2 - y1
    d = pd.read_csv(withgrids, header=[0, 1, 2], index_col=0)
    sub = d[d.columns.get_level_values(0)[0]]
    bp = next(b for b in sub.columns.get_level_values(0).unique() if "grid_number" in sub[b].columns)
    b = sub[bp]
    good = np.flatnonzero((b.likelihood > 0.95).to_numpy() & b.grid_number.notna().to_numpy() & (np.arange(len(b)) > len(b) * 0.2))
    fi = int(good[len(good) // 2]) if len(good) else len(b) // 2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    img = frame[y1:y2, x1:x2].copy()
    ch = H / n_cols

    def cell_box(k):
        c, r = divmod(k, n_cols)
        xa = fit["x_right"] - (c + 1) * fit["cell_px"] if rotate_180 else fit["x0"] + c * fit["cell_px"]
        ya = H - (r + 1) * ch if rotate_180 else r * ch
        return (int(xa), int(ya)), (int(xa + fit["cell_px"]), int(ya + ch))

    over = img.copy()
    for k in entry:
        cv2.rectangle(over, *cell_box(k), (255, 120, 0), -1)
    for k in target:
        cv2.rectangle(over, *cell_box(k), (0, 200, 255), -1)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for i in range(n_cols + 1):
        cv2.line(img, (int(i * W / n_cols), 0), (int(i * W / n_cols), H), (0, 0, 255), 1)
    for k in range(n_cols * n_cols):
        p0, p1 = cell_box(k)
        cv2.rectangle(img, p0, p1, (0, 255, 0), 1)
        cv2.putText(img, str(k), (p0[0] + 4, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    if pd.notna(b.x.iloc[fi]):
        bx, by = int(b.x.iloc[fi]), int(b.y.iloc[fi])
        cv2.circle(img, (bx, by), 9, (0, 0, 255), -1)
        cv2.putText(img, f"{bp} cell {int(b.grid_number.iloc[fi])}", (bx + 12, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{label}  frame {fi}  green = aligned grid, red = crop grid", (8, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def align_project_grids(config, cohort_metadata, n_cols=10, rotate_180=True, max_resid_px=8.0,
                        cell_tolerance=0.15, min_walls=6, force=False):
    """
    Align every session's grid columns to its maze walls; skip sessions already aligned.

    Videos come from a cohort metadata column ``Video Path`` (absolute, or relative to the
    folder above ``compass_analysis``). Without that column nothing is done (crop grids kept).
    A session without a readable video, or whose fit fails
    the checks (fewer than ``min_walls`` walls, a wall more than ``max_resid_px`` off the fit, or
    a cell width more than ``cell_tolerance`` away from crop width / n_cols), keeps its crop grid.

    Returns (alignment table, changed): ``changed`` is True when any frame's cell differs from
    what was on disk before this call, i.e. 01's preprocessed CSVs need rebuilding.
    """
    proj = Path(config["project_path_full"])
    if not any(c in cohort_metadata.columns for c in VIDEO_COLUMNS):
        print("grid alignment: no Video Path column in the metadata -- keeping the crop grids")
        return pd.DataFrame(), False
    out = proj / ALIGN_DIRNAME
    orig = out / "withGrids_original"
    orig.mkdir(parents=True, exist_ok=True)
    table_path = out / "alignment.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    # aligned file hash per session: a different hash means 00 rewrote _withGrids.csv from the crop grid
    aligned_hash = ({r.session: r.get("withgrids_sha1") for _, r in table.iterrows() if r.status == "aligned"}
                    if len(table) else {})
    dlc_dir = Path(config.get("dlc_results_path") or proj / "dlc_results")
    scorer = config.get("dlc_scorer", "")
    zones = config.get("zones") or {}
    entry, target = zones.get("entry_zone", []), zones.get("target_zone", [])

    rows, changed = [], False
    for _, m in cohort_metadata.iterrows():
        session = f"Session{int(m['session']):04d}"
        if session not in config.get("session_names", [session]):
            continue
        row = dict(session=session)
        wg = dlc_dir / f"{session}_withGrids.csv"
        if not wg.exists():
            rows.append({**row, "status": "no _withGrids.csv"}); continue
        cur_hash = _sha1(wg)
        prev_hash = aligned_hash.get(session)
        if isinstance(prev_hash, str) and prev_hash == cur_hash and not force:
            continue
        video = _video_for_session(m, proj)
        if video is None or not video.exists():  # crop grid kept: _withGrids.csv untouched
            status = f"crop grid kept (video not found: {video})" if video else "crop grid kept (no video path)"
            rows.append({**row, "status": status})
            print(f"{session}: {status}", flush=True)
            continue
        if not (orig / wg.name).exists() or (isinstance(prev_hash, str) and prev_hash != cur_hash):
            shutil.copy2(wg, orig / wg.name)  # fresh crop-grid file (first run, or 00 re-assigned grid numbers)
        crop = _crop_for_session(dlc_dir, session, scorer, m)
        W = crop[1] - crop[0]
        fit = fit_grid_to_walls(read_background(video, crop), n_cols=n_cols)
        row.update(video=str(video), crop_x1=crop[0], crop_x2=crop[1], crop_y1=crop[2], crop_y2=crop[3],
                   maze_left_px=round(fit["x0"], 1), maze_right_px=round(fit["x_right"], 1),
                   cell_px=round(fit["cell_px"], 1), crop_cell_px=round(W / n_cols, 1),
                   n_walls=fit["n_walls"], max_resid_px=round(fit["max_resid_px"], 1))
        bad = (fit["n_walls"] < min_walls or fit["max_resid_px"] > max_resid_px
               or abs(fit["cell_px"] / (W / n_cols) - 1) > cell_tolerance)
        if bad:
            # keep the crop grid: restore the original file
            if Path(wg).read_bytes() != (orig / wg.name).read_bytes():
                shutil.copy2(orig / wg.name, wg)
                changed = True
            fit_used = dict(x0=0.0, x_right=float(W), cell_px=W / n_cols)
            row.update(status="fit rejected: crop grid kept", frames_reassigned_pct=0.0)
        else:
            fit_used = fit
            n, ch_src, ch_cur = _reassign_columns(orig / wg.name, wg, fit, n_cols, rotate_180)
            changed |= ch_cur > 0
            row.update(status="aligned", frames_reassigned_pct=round(100 * ch_src / max(n, 1), 1), withgrids_sha1=_sha1(wg))
        _draw_overlay(video, crop, wg, fit_used, n_cols, rotate_180, entry, target,
                      out / f"{session}_alignment.jpg", f"{session} mouse {m.get('mouseid', '')}")
        rows.append(row)
        print(f"{session}: {row['status']}, {row.get('frames_reassigned_pct', 0)}% frames re-assigned", flush=True)

    if rows:
        new = pd.DataFrame(rows)
        table = pd.concat([table[~table.session.isin(new.session)] if len(table) else table, new], ignore_index=True)
        table.sort_values("session").to_csv(table_path, index=False)
    else:
        print("grid alignment: all sessions already aligned, skipped")
    return table, changed
