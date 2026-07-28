# =============================================================================
# PRE-FIXED VALUES (DO NOT EDIT)
# =============================================================================
#
# Maze: 10x10 grid, single serpentine (hairpin) corridor, no branching.
# FID = col*10 + row, 0-indexed col/row, col0 = leftmost, row0 = top --
# matching the column-major grid numbering produced by
# get_grid_coordinates()/batch_create_grids().
#
# Path runs: start (col0,row0 = FID0, top-left) -> down col0 -> turn T1
# (col0<->col1, bottom) -> up col1 -> turn T2 (col1<->col2, top) -> ... ->
# turn T7 (col6<->col7, bottom) -> up col7 -> reward zone (turn col7<->col8,
# top, FIDs 70,71,80,81) -> down col8 -> turn T9 (col8<->col9, bottom) -> up
# col9 -> overshoot terminal (col9, top, FIDs 90,91 -- no further column).
# Turn numbers count UP from start (T1) to the last turn before the
# overshoot terminal (T9); the reward zone sits at the 8th of 9 turns
# (second-to-last). Every turn is a 2x2 open block (4 cells spanning the two
# columns it connects), not a single-cell pivot, so all 4 cells of a turn
# are mutually adjacent.

import pandas as pd
from importlib.resources import files


# -------------------- REGION---------------------------#
# Map Grid Nodes to Regions
ENTRY_ZONE = [0, 1]  # top-left corner + its one physical neighbor (col0, rows 0-1) --
# grown from just [0] because entry_zone's single cell is easy for sparse
# tracking to miss entirely for a whole session, which broke bout
# segmentation (see assign_bout_indices_from_entry_zone())
TARGET_ZONE = [70, 71, 80, 81]  # reward zone: turn between col7/col8

# Turn (corner) blocks between start and the reward zone, numbered T1 (first
# turn leaving start) through T7 (last turn before the reward zone). Each is
# a 2x2 open room spanning the two columns it connects.
CORNER_REWARD = [
    8, 9, 18, 19,      # T1: turn col0 <-> col1 (bottom)
    10, 11, 20, 21,    # T2: turn col1 <-> col2 (top)
    28, 29, 38, 39,    # T3: turn col2 <-> col3 (bottom)
    30, 31, 40, 41,    # T4: turn col3 <-> col4 (top)
    48, 49, 58, 59,    # T5: turn col4 <-> col5 (bottom)
    50, 51, 60, 61,    # T6: turn col5 <-> col6 (top)
    68, 69, 78, 79,    # T7: turn col6 <-> col7 (bottom)
]

# Straight (non-corner) corridor cells between start and the reward zone.
NONDECISION_REWARD = [
    2, 3, 4, 5, 6, 7,              # col0 (below the entry zone)
    12, 13, 14, 15, 16, 17,        # col1
    22, 23, 24, 25, 26, 27,        # col2
    32, 33, 34, 35, 36, 37,        # col3
    42, 43, 44, 45, 46, 47,        # col4
    52, 53, 54, 55, 56, 57,        # col5
    62, 63, 64, 65, 66, 67,        # col6
    72, 73, 74, 75, 76, 77,        # col7 (up to the reward-zone turn)
]

REWARD_PATH = NONDECISION_REWARD + CORNER_REWARD

# Turn (corner) block and straight cells past the reward zone: anything after
# the reward zone to the end (col8, col9, and the overshoot terminal) --
# corridor that continues to physically exist past the target but isn't part
# of the intended route.
CORNER_OVERSHOOT = [
    88, 89, 98, 99,  # T9: turn col8 <-> col9 (bottom; last turn before the terminal)
]
NONDECISION_OVERSHOOT = [
    82, 83, 84, 85, 86, 87,        # col8
    92, 93, 94, 95, 96, 97,        # col9
    90, 91,                        # overshoot terminal (col9, top) -- straight, not a
                                   # corner: no paired column beyond it, so no cross-edges
]

OVERSHOOT = NONDECISION_OVERSHOOT + CORNER_OVERSHOOT
# ------------------------------------------------------#

REGION_NAMES = {
    "entry_zone": "Entry Zone",
    "target_zone": "Target Zone",
    "reward_path": "Reward Path",
    "overshoot": "Overshoot",
}

# -------CHOOSE REGION NAMES (KEY), MAPPED TO LIST OF GRID NODES IN THAT REGION (VALUE)--------#
REGION_MAPPING = {
    "entry_zone": ENTRY_ZONE,
    "target_zone": TARGET_ZONE,
    "reward_path": REWARD_PATH,
    "overshoot": OVERSHOOT,
}

# -------CHOOSE REGION NAMES (KEY), MAPPED TO LENGTH OF GRID NODES IN THAT REGION (VALUE)--------#
REGION_LENGTHS = {
    "entry_zone": len(ENTRY_ZONE),
    "target_zone": len(TARGET_ZONE),
    "reward_path": len(REWARD_PATH),
    "overshoot": len(OVERSHOOT),
}

# ----------------NODE-TYPES-------------------------#
# No branching anywhere in this maze, so there are no "decision" node types
# (a decision requires a genuine multi-way choice of next cell). Every turn
# is a forced U-turn -> classified as "corner", not "decision".
NODE_NAMES = {
    "entry_zone": "Entry Zone",
    "target_zone": "Target Zone",
    "corner_reward": "Corner (Reward Path)",
    "nondecision_reward": "Straight (Reward Path)",
    "corner_overshoot": "Corner (Overshoot)",
    "nondecision_overshoot": "Straight (Overshoot)",
}

ENTRY_ZONE_NODES = ENTRY_ZONE
TARGET_ZONE_NODES = TARGET_ZONE

NODE_TYPE_MAPPING = {
    "entry_zone": ENTRY_ZONE_NODES,
    "target_zone": TARGET_ZONE_NODES,
    "corner_reward": CORNER_REWARD,
    "nondecision_reward": NONDECISION_REWARD,
    "corner_overshoot": CORNER_OVERSHOOT,
    "nondecision_overshoot": NONDECISION_OVERSHOOT,
}

# ----------------PACKAGE DATA-------------------------#
# Load adjacency matrix from package data
# Tolerance is 3 grid-steps per frame (not 4): cells are 1.2x larger than the
# old 12x12 maze's (10x10 over the same physical arena), so the same
# physical speed covers proportionally fewer new-grid-cells per frame.
# Re-validate this against real tracking data once available.
ADJACENCY_MATRIX = pd.read_csv(
    files("compass_labyrinth.data").joinpath("3step_adjacency_matrix.csv"),
)

NODES = range(0, 100)
