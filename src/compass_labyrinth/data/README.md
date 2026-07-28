# Resources

This folder contains reusable static files and reference maps required by various modules. These are not raw data files, but configuration-driven or model-specific resources that guide downstream processing and analysis.

---

## Contents

### `3step_adjacency_matrix.csv`
- Binary matrix that defines valid transitions between neighboring grid nodes in the current 10x10 maze — where only up to 3 connections/steps in 0.2 sec (1 frame) are allowed, generated from the maze's corridor graph (serpentine lane chain + 2x2 open turn rooms). See `constants.py` for how the tolerance was derived from cell size.
- Used within data preprocessing

### `4step_adjacency_matrix.csv`, `Value_Function_perGridCell.csv`
- Legacy resources for the old 12x12 (144-cell) maze. Not used by `constants.py` for the current 10x10 maze — kept for reference only. `Value_Function_perGridCell.csv` was used by CoMPASS Level 2, which hasn't been re-derived for the new maze yet.
