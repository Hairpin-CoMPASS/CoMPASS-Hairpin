"""Report text config for the hairpin-maze figure report.

Maps each pipeline's figure folders (under `figures/<pipeline_key>/`) to a
display title and description. Folder names and order here must be kept in
sync with the `figures_dirname=` calls in `tutorials/02b_*.ipynb`
(task_performance) and `tutorials/05b_*.ipynb` (level_1) -- this is the
project-specific adapter the report builder reads instead of hard-coding
anything into the report logic itself.

Each step's description is `{"summary": "<one short line>", "items": [(label,
text_or_bullets), ...]}`, rendered as a lead-in sentence plus a label/value
list (e.g. "X axis", "Y axis", "How to read", "Statistics") -- `text` can be
a plain string or a list of strings for a bullet list. Keep "summary" short
and self-contained; it is read on its own in the step-heading tooltip.

A step whose figures come from more than one genotype comparison is detected
by the report from the comparison key (e.g. "wt_nlf") each file's name
already carries, and rendered with a "Comparison" selector instead of
showing all of them at once. The set of comparison keys and their labels for
*this* project comes from `config.yaml`'s `comparisons`/`genotype_sets`
(read by `_load_comparisons()` in build.py), not from anything in this file
-- so a future project with different genotype labels needs no changes here.
"""

REPORT_TITLE = "E3E4_Hairpin_01"

# Reused verbatim across several steps so the wording of a shared concept
# doesn't drift between them.
# Level 1 maze heatmaps: the same maps split by when in the session the frames are
TIME_BINS_SELECTOR = {"label": "Time bins", "none_label": "All frames", "options": [
    ("pct_bins", "% of session"), ("frame_bins", "Frames")]}
DIRECTION_SELECTOR = {"label": "Direction", "options": [
    ("going_in", "Going in"), ("coming_back", "Coming back"), ("backtrack", "Backtrack"), ("reentry", "Re-entry"),
    ("change", "Coming back \u2212 going in")]}
TRAVEL_ITEM = ("Direction", [
    "chosen with the Direction selector above; Plot 5's phases grouped by which way the mouse travels.",
    "going in = outbound + reward-zone frames moving deeper + overshoot forward.",
    "coming back = return + reward-zone frames moving back toward the entry + overshoot backward.",
    "backtrack = backing up on the path before reaching the reward (or the bout's furthest point, if it never got there); re-entry = heading back in on the path after it. Both are sparse, so many cells are grey.",
    "each reward-zone frame is in exactly one map, by the direction it is moving; pauses go with the movement they interrupt.",
])
TIME_BINS_ITEM = [
    "chosen with the Time bins selector above. All frames = the maps above.",
    "% of session: each session's analyzed frames split into 5 equal parts in frame order (0-20%, ..., 80-100%), so every mouse fills every column.",
    "Frames: analyzed frames 0-10k, 10k-20k, ..., 40k-50k of each session; a session shorter than 50k frames only fills the earlier columns (\"n=X of Y\" under a panel = mice with frames in that bin), and frames past 50k are left out (about a quarter of this dataset's frames).",
    "one column per time bin, one row per genotype group (6a, 6b, 7, 7b: one figure per direction; 5f: backtrack rows, then re-entry rows); same color scale as the all-frames maps.",
    "difference rows at the bottom: second group minus first per bin, cell by cell (same pairs as the all-frames maps: B − A, or E3NLF − E3, E4NLF − E4, E4 − E3, E4NLF − E3NLF), on a blue-white-red scale centered at 0 with one limit per figure (95th percentile of its differences); a cell is grey when either group is missing there.",
    "per-mouse maps need >= 10 frames (6a/6b, 7, 7b) or >= 3 passes (5f) per mouse in that cell and bin (a bin has about a fifth of the frames); 1-1f pool the group's frames, no minimum.",
]


COMPARISON_BULLETS = [
    "the 4 individual genotypes (E3, E3NLF, E4, E4NLF) side by side.",
    "pooled WT (E3 + E4) vs. NLF (E3NLF + E4NLF).",
    "pooled APOE3 (E3 + E3NLF) vs. APOE4 (E4 + E4NLF).",
]
FDR_PAIRWISE_BIN = (
    "FDR-corrected pairwise t-tests between genotypes, run separately in each time bin, "
    "in the Statistics table below."
)

PIPELINES = {
    "task_performance": {
        "name": "Task performance",
        "steps": [
            {
                "folder": "exclusion_criteria",
                "title": "Exclusion criteria",
                "summary": "Target-zone usage vs. session length for every session, used to flag and drop unreliable sessions before any comparison figure below runs.",
                "items": [
                    ("X axis", "total number of tracked frames in the session."),
                    ("Y axis", "target-zone usage: the target zone's share of area-normalized occupancy among entry zone, reward path, overshoot and target zone, averaged over the session's 10,000-frame bins."),
                    ("Normalization", "YES, by zone size -- each region's frame count is divided by its number of grid cells, then the 4 regions are rescaled to sum to 1 per bin. So the 40% threshold compares time-per-cell, not raw time."),
                    ("Points", "one per session, colored by genotype; the second panel marks sessions that fail either threshold with an X."),
                    ("Exclusion thresholds", "target-zone usage below 40%, or fewer than 30,000 tracked frames -- either one drops the session from every figure below."),
                    ("Statistics", "none (a session-filtering step, not a group comparison)."),
                ],
            },
            {
                "folder": "plot1_heatmap",
                "title": "1 Heatmap",
                "summary": "Spatial heatmap of time spent in each maze zone (excluding the entry corridor) across time bins, with every genotype group overlaid in one figure.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("X axis", "individual sessions, one column per valid session for that genotype."),
                    ("Y axis", "maze zones, ordered top (entry corridor) to bottom (reward zone)."),
                    ("Panels", "one row of panels per time bin (5,000-frame bins, labeled by frame range, up to 70,000 frames), one column of panels per genotype in the chosen comparison."),
                    ("Group mean", "rightmost panel of each row: one column per genotype group, each cell the mean across that group's sessions for that zone (value printed on the cell)."),
                    ("Color", "occupancy density: the zone's share of that session's time in that bin, per grid cell."),
                    ("Normalization", "YES, by zone size -- each zone's frame count is divided by its number of grid cells, then each session's column (one bin) is rescaled so the zones sum to 1. Zones of different sizes are directly comparable; 1/(number of zones) everywhere would mean time spread in proportion to zone size."),
                    ("Statistics", "none on this figure (an occupancy map); see Plot 3 for region usage over time with significance testing."),
                ],
            },
            {
                "folder": "plot1b_heatmap_pct",
                "title": "1b Heatmap (% of session)",
                "summary": "Same spatial heatmap as Plot 1, but binned by percentage of each session's own length instead of a fixed frame count, so short and long sessions contribute equally.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("X axis", "individual sessions, one column per valid session for that genotype."),
                    ("Y axis", "maze zones, ordered top (entry corridor) to bottom (reward zone)."),
                    ("Panels", "one row of panels per 20%-of-session bin (first 20%, next 20%, ...), one column of panels per genotype in the chosen comparison."),
                    ("Group mean", "rightmost panel of each row: one column per genotype group, each cell the mean across that group's sessions for that zone."),
                    ("Color", "occupancy density: the zone's share of that session's time in that bin, per grid cell."),
                    ("Normalization", "YES, by zone size (as Plot 1: frames ÷ grid cells, zones rescaled to sum to 1 per session per bin), and by session length -- bins are percentages of each session's own frames."),
                    ("Statistics", "none on this figure."),
                ],
            },
            {
                "folder": "plot2_entropy",
                "title": "2 Entropy over time",
                "summary": "Shannon entropy of region occupancy across time bins -- how evenly a mouse spreads its time across zones vs. concentrating in a few.",
                "items": [
                    ("X axis", "cumulative frames in the session (time bins)."),
                    ("Y axis", "Shannon entropy (log base 2) of zone occupancy in that bin; higher = time spread evenly across zones, lower = concentrated in fewer zones. Maximum = log2(number of zones)."),
                    ("Normalization", "YES, by zone size -- entropy is computed on the area-normalized occupancy (frames ÷ grid cells, as Plot 1), so the maximum means time spent in proportion to each zone's size, not equal raw time in every zone."),
                    ("Lines", "one per genotype (mean ± SE); a second, zoomed-in panel is included and cuts off once the per-genotype sample size gets thin."),
                    ("How to read", "a genotype with lower entropy is spending disproportionate time in a subset of regions rather than exploring broadly."),
                    ("Statistics", "repeated-measures ANOVA (time bin as the within-subject factor) printed above the figure; per-pair mixed models testing the Bin × Genotype interaction, plus " + FDR_PAIRWISE_BIN),
                ],
            },
            {
                "folder": "plot3_all_regions_usage",
                "title": "3 All regions overlay",
                "summary": "All region-usage proportions (zones excluding the entry corridor) overlaid on a single plot across time bins, comparing pooled WT vs. NLF.",
                "items": [
                    ("X axis", "cumulative frames in the session (time bins), capped at the longest common session length."),
                    ("Panels", "one subplot per region."),
                    ("Y axis", "the region's share of area-normalized occupancy in that bin (mean ± SE across sessions)."),
                    ("Normalization", "YES, by zone size -- frames ÷ grid cells, regions rescaled to sum to 1 per session per bin (same values as Plot 1), so regions of different sizes are comparable."),
                    ("Lines", "one per genotype group within each region's subplot."),
                    ("How to read", "compare lines WITHIN a subplot (same region, between genotype groups), and subplots to each other (which regions a genotype favors over the session)."),
                    ("Statistics", "mixed-effects model (Bin × Genotype) run per region; per-region " + FDR_PAIRWISE_BIN),
                ],
            },
            {
                "folder": "plot3b_all_regions_usage_pct",
                "title": "3b All regions overlay (%)",
                "summary": "Same overlay as Plot 3, with time expressed as percentage of each session's own length instead of a fixed frame count.",
                "items": [
                    ("X axis", "% of the way through the session (0-100%)."),
                    ("Panels", "one subplot per region."),
                    ("Y axis", "the region's share of area-normalized occupancy in that bin (mean ± SE across sessions)."),
                    ("Normalization", "YES, by zone size (as Plot 3) and by session length -- bins are percentages of each session's own frames."),
                    ("Lines", "one per genotype group within each region's subplot."),
                    ("How to read", "compare lines WITHIN a subplot (same region, between genotype groups), and subplots to each other."),
                    ("Statistics", "none on this overlay; see Plot 3 for the frame-based version's per-region significance testing."),
                ],
            },
            {
                "folder": "plot4_success_rate",
                "title": "4 Success rate (cumulative)",
                "summary": "Cumulative percentage of bouts that reached the target zone before returning to the entry zone, WT vs. NLF.",
                "items": [
                    ("X axis", "genotype group (WT, NLF)."),
                    ("Y axis", "% of bouts, over the whole session, that reached the target zone."),
                    ("Normalization", "by bout count, not zone size -- successful bouts ÷ valid bouts (longer than 20 frames) in that session, so sessions with more or fewer bouts are comparable."),
                    ("Points", "bar = group mean ± SE; one dot per session overlaid."),
                    ("How to read", "a higher bar means a larger share of that genotype's attempts reached the reward."),
                    ("Statistics", "Welch's t-test (unequal variance) between genotypes, printed above the figure and in the Statistics table below."),
                ],
            },
            {
                "folder": "plot5_time_based_success",
                "title": "5 Success rate (time bins)",
                "summary": "Percentage of successful bouts within sequential time bins, WT vs. NLF.",
                "items": [
                    ("X axis", "cumulative frames in the session (time bins)."),
                    ("Y axis", "% of bouts in that bin that reached the target zone."),
                    ("Normalization", "by bout count, not zone size -- successful ÷ valid bouts (longer than 19 frames) within each 5,000-frame bin; a bout spanning two bins counts in both."),
                    ("Lines", "one per genotype (mean ± SE)."),
                    ("How to read", "a rising line over the session indicates the genotype is learning the route; compare genotypes within the same bin to see who improves faster or plateaus higher."),
                    ("Statistics", "repeated-measures ANOVA is attempted but commonly fails here (\"Between subject effect not yet supported\") -- a statsmodels limitation for this between+within design, not a data problem; the reliable result is the FDR-corrected pairwise comparison per bin in the Statistics table below."),
                ],
            },
            {
                "folder": "plot5b_time_based_success_pct",
                "title": "5b Success rate (time bins, %)",
                "summary": "Same binned success rate as Plot 5, with time expressed as percentage of each session's own length instead of a fixed frame range.",
                "items": [
                    ("X axis", "% of the way through the session (0-100%)."),
                    ("Y axis", "% of bouts in that bin that reached the target zone."),
                    ("Normalization", "by bout count (successful ÷ valid bouts in the bin) and by session length (bins are percentages of each session's own frames); not by zone size."),
                    ("Lines", "one per genotype (mean ± SE)."),
                    ("How to read", "a rising line over the session indicates the genotype is learning the route; compare genotypes within the same bin."),
                    ("Statistics", "same caveat as Plot 5; the FDR-corrected pairwise comparison per bin in the Statistics table below is the reliable result."),
                ],
            },
            {
                "folder": "plot6_zone_occupancy_velocity",
                "title": "6 Zone occupancy & velocity",
                "summary": "Per-bout occupancy fraction of reward-adjacent zones, smoothed and fit with an exponential curve across bouts.",
                "items": [
                    ("X axis", "sequential bout number."),
                    ("Y axis", "fraction of that bout's frames spent in the zone (colored lines), and normalized bout velocity (black line)."),
                    ("Normalization", "Zone occupancy: NOT normalized by zone size -- raw frames in the zone ÷ frames in the bout, so a bigger zone collects a larger fraction just by its size. Compare the same zone across bouts or genotypes, not one zone's level against another's. Velocity: YES, rescaled to 0-1 (robust scaling, then a quantile transform across that group's bouts) so it can share the axis -- its shape is meaningful, not its absolute value."),
                    ("Lines", "one color per zone (smoothed mean with a confidence band), dashed = exponential fit; one panel per genotype group, plus a successful-bouts-only figure."),
                    ("How to read", "a line decaying toward a lower asymptote means the mouse learns to move through that zone faster (or avoid lingering in it) as bouts accumulate."),
                    ("Statistics", "none on these figures directly; exponential fit parameters are in the Data sheets."),
                ],
            },
            {
                "folder": "plot6b_zone_occupancy_velocity_pct",
                "title": "6b Zone occupancy & velocity (%)",
                "summary": "Same per-zone occupancy curves as Plot 6, with each bout's position expressed as a percentage of that session's own total bout count instead of a raw bout number.",
                "items": [
                    ("X axis", "% of the way through that session's own bouts (0-100%)."),
                    ("Y axis", "fraction of that bout's frames spent in the zone (colored lines), and normalized bout velocity (black line)."),
                    ("Normalization", "Zone occupancy: NOT normalized by zone size (as Plot 6). Bout position: YES, by each session's own bout count. Velocity: YES, rescaled to 0-1 (as Plot 6)."),
                    ("Lines", "one color per zone (smoothed mean with a confidence band), dashed = exponential fit; one panel per genotype group, plus a successful-bouts-only figure."),
                    ("How to read", "lets mice with very different total bout counts be compared at the same relative point in their session."),
                    ("Statistics", "none on these figures directly; exponential fit parameters are in the Data sheets."),
                ],
            },
        ],
    },
    "level_1": {
        "name": "Level 1",
        "steps": [
            {
                "folder": "plot1_heatmap",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "zone_set_none": True,  # "none" = the per-cell map; zone sets = the zone-pooled versions
                "title": "1 HMM state heatmap",
                "summary": "Spatial heatmap of HMM state proportion per maze grid cell -- the Slow / Surveillance state's (HMM state 1) share of frames -- with every genotype group overlaid in one figure.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Zone set", "none (default) = the per-cell map; choosing a zone set (e.g. 10_rows) shows the same measure pooled per zone instead -- each zone colored by its value over all of the group's frames there and labeled with it; cells outside the set are hatched grey."),
                    ("Layout", "the actual 10x10 hairpin maze grid, one subplot per genotype in the chosen comparison; the reward/target zone is outlined."),
                    ("Color", "proportion of that cell's frames classified as HMM state 1 (Slow / Surveillance, after merging in state 0/Stationary frames -- those excluded from HMM fitting because their step length was too small to reliably estimate a turning angle); values are also annotated directly on each cell."),
                    ("Normalization", "a ratio within each cell (state-1 frames ÷ all frames in that cell), so it doesn't depend on how long the mouse spent there; every cell is the same size, so zone size doesn't apply. Frames are pooled across all of a group's sessions (a session with more frames in a cell weighs more), not averaged per session."),
                    ("How to read", "darker (higher) cells are where the mouse spent more of its time surveilling (slow, high-turning-angle movement) rather than moving straight through (Fast / Ambulatory, state 2); compare the SAME cell across genotype panels."),
                    ("Statistics", "none on this figure; see Plot 2 for state probability by zone with significance testing."),
                ],
            },
            {
                "folder": "plot1b_velocity_heatmap",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "zone_set_none": True,  # "none" = the per-cell map; zone sets = the zone-pooled versions
                "title": "1b Velocity heatmap",
                "summary": "Spatial heatmap of mean velocity per maze grid cell, with every genotype group overlaid in one figure.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Zone set", "none (default) = the per-cell map; choosing a zone set (e.g. 10_rows) shows the same measure pooled per zone instead -- each zone colored by its value over all of the group's frames there and labeled with it; cells outside the set are hatched grey."),
                    ("Layout", "the actual 10x10 hairpin maze grid, one subplot per genotype in the chosen comparison; the reward/target zone is outlined."),
                    ("Color", "mean velocity of frames in that grid cell; values are also annotated directly on each cell."),
                    ("Normalization", "NOT normalized -- raw velocity (tracking distance per frame x 5 fps, in tracking units per second), averaged over all frames in the cell pooled across the group's sessions. Every cell is the same size, so zone size doesn't apply."),
                    ("How to read", "darker (higher-velocity) cells are where the mouse moved through fastest; compare the SAME cell across genotype panels."),
                    ("Statistics", "none on this figure; see Plot 2b for velocity by zone with significance testing."),
                ],
            },
            {
                "folder": "plot1c_sternum_speed",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "title": "1c Sternum speed",
                "summary": "Mean speed of the sternum relative to the maze, per grid cell, by genotype group -- the same measure as Plot 1b, for the sternum instead of the belly.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Layout", "as Plot 1b: one maze panel per genotype group, same 0-300 color scale; frames assigned to cells by the belly."),
                    ("How it's calculated", "same rules as the pipeline's velocity: only frames with sternum likelihood > 0.6 are used (94% of analyzed frames), and speed = distance between consecutive kept rows x 5 fps."),
                    ("Sternum vs. belly", "the sternum is detected on more frames than the belly, but its position is less precise (median frame-to-frame jitter 5.4 vs. 3.6 px, higher in every session), and it moves with the shoulders and head; so its speed runs higher (median 47 vs. 25) partly from tracking noise. Belly speed (Plot 1b) is the cleaner measure of body locomotion."),
                    ("Normalization", "NOT normalized -- raw speed; an average, so cell size doesn't inflate it."),
                    ("Statistics", "none on this figure."),
                ],
            },
            {
                "folder": "plot1d_nose_speed_allocentric",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "title": "1d Nose speed (allocentric)",
                "summary": "Mean speed of the nose relative to the maze, per grid cell, with every genotype group side by side -- like Plot 1b, but for the nose.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Layout", "the 10x10 hairpin maze grid, one panel per genotype group; the reward/target zone is outlined. Frames are assigned to cells by the belly, the point the whole pipeline tracks."),
                    ("Color", "mean nose speed of the frames in that cell (tracking units per second); values are annotated on each cell."),
                    ("How it's calculated", "nose positions come from the DLC output the pipeline was built from (dlc_results/SessionXXXX_withGrids.csv), joined by frame number (s_no). Same rules as the pipeline's velocity: only frames with nose likelihood > 0.6 are used (others dropped, not interpolated), and speed = distance between consecutive kept rows x 5 fps."),
                    ("How to read", "the nose moves with the body and with the head, so cells where nose speed is high but body speed (Plot 1b) is low are where mice move their head a lot while staying in place."),
                    ("Normalization", "NOT normalized -- raw speed; an average over the cell's frames pooled across the group's sessions, so cell size doesn't inflate it."),
                    ("Statistics", "none on this figure."),
                ],
            },
            {
                "folder": "plot1e_nose_speed_egocentric",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "title": "1e Nose speed (egocentric)",
                "summary": "Mean speed of the nose relative to the mouse's own body -- head movement with the body's movement and turning removed -- per grid cell, by genotype group.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Layout", "as Plot 1d: one maze panel per genotype group, frames assigned to cells by the belly."),
                    ("Body frame", "origin = sternum; forward axis = from the tail base to the sternum. Each frame, the nose's position relative to the sternum is rotated into this frame (forward / sideways), so walking or turning on its own doesn't move it -- only the head moving relative to the body does. The sternum is used rather than the belly because the tail base -> sternum axis is about twice as long (~96 vs. ~44 px) and the sternum sits closer to the nose (~49 vs. ~97 px), so tracking wobble tilts the frame much less."),
                    ("Color", "mean speed of the nose in that body frame (tracking units per second); values are annotated on each cell."),
                    ("How it's calculated", "same rules as the pipeline's velocity: only frames where nose, sternum and tail base all have likelihood > 0.6 are used (about 69% of analyzed frames), and speed = distance between consecutive kept rows' body-frame nose positions x 5 fps. Tracking jitter in the three points adds some speed even when the head is still."),
                    ("How to read", "higher = more head scanning (looking around) in that cell, independent of how fast the mouse is moving through it; compare with Plot 1/2 surveillance."),
                    ("Normalization", "NOT normalized -- raw speed; an average, so cell size doesn't inflate it."),
                    ("Statistics", "none on this figure."),
                ],
            },
            {
                "folder": "plot1f_head_angle_absolute",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "title": "1f Head angle (absolute)",
                "summary": "How far the head is turned away from the body axis, per grid cell, by genotype group -- either side counts.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. Two groups: B \u2212 A centered underneath. Four groups: the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Layout", "as Plot 1d: one maze panel per genotype group; frames assigned to cells by the belly."),
                    ("Head angle", "the direction of the nose from the sternum, relative to the body axis (tail base -> sternum): 0° = head straight ahead, 90° = turned fully to the side. Absolute = the size of the turn, ignoring which side."),
                    ("Color", "mean absolute head angle of the cell's frames (degrees)."),
                    ("How it's calculated", "only frames where nose, sternum and tail base all have likelihood > 0.6 (about 69% of analyzed frames); the angle is computed per frame (no speeds involved)."),
                    ("How to read", "higher = the head is turned further from the body axis there -- looking around, or the body bending through a turn; it is highest at the U-turns. Across all frames the head is turned further during the Slow / Surveillance state (median 20° vs. 12°)."),
                    ("Normalization", "NOT normalized -- an angle in degrees, averaged over the cell's frames pooled across the group's sessions."),
                    ("Statistics", "none on this figure."),
                ],
            },
            {
                "folder": "plot2_state_probability",
                "title": "2 State probability by zone",
                "summary": "Boxplot of Slow / Surveillance state probability across the 9 individual hairpin turns, compared by genotype.",
                "items": [
                    ("X axis", "the 9 hairpin-turn zones."),
                    ("Y axis", "fraction of a session's frames in that zone classified as the Slow / Surveillance state (HMM state 1, with Stationary frames merged in) -- one value per session."),
                    ("Normalization", "a ratio within each zone (state-1 frames ÷ all of that session's frames in the zone), so it is independent of zone size and of how long the mouse spent in the zone -- no area normalization needed."),
                    ("Boxes", "one per genotype at each zone: the line is the median session, the box spans the middle 50% of sessions (25th-75th percentile), and the whiskers (range bars) reach the furthest session within 1.5x the box height; sessions beyond that are drawn as individual points (outliers)."),
                    ("How to read", "a genotype/zone combination with a higher box spends more time surveilling (slow, high-turning-angle movement) there instead of moving straight through."),
                    ("Statistics", "pairwise t-tests between genotypes within each zone, sorted by p-value, in the Statistics table below."),
                ],
            },
            {
                "folder": "plot2b_velocity",
                "title": "2b Velocity by zone",
                "summary": "Boxplot of mean velocity across 5 column zones, compared by genotype.",
                "items": [
                    ("X axis", "the 5 column zones."),
                    ("Y axis", "mean velocity within that zone -- one value per session."),
                    ("Normalization", "NOT normalized -- raw velocity (tracking units per second) averaged over the session's frames in that zone; an average, so zone size doesn't inflate it."),
                    ("Boxes", "one per genotype at each zone: the line is the median session, the box spans the middle 50% of sessions (25th-75th percentile), and the whiskers (range bars) reach the furthest session within 1.5x the box height; sessions beyond that are drawn as individual points (outliers)."),
                    ("How to read", "a genotype/zone combination with a lower box is moving more slowly -- consistent with more surveillance/pausing there."),
                    ("Statistics", "pairwise t-tests between genotypes within each zone, sorted by p-value, in the Statistics table below."),
                ],
            },
            {
                "folder": "plot3_temporal_median",
                "title": "3 State probability over time",
                "summary": "Median Surveillance-state probability over time, one curve per zone (corners, columns, reward), overlaid by genotype.",
                "items": [
                    ("X axis", "cumulative frames in the session (time bins)."),
                    ("Y axis", "for each session and time bin: the median, across the zone's grid cells, of each cell's Slow / Surveillance (HMM state 1) proportion; points are the mean ± SE of those per-session medians."),
                    ("Normalization", "a ratio within each cell (state-1 frames ÷ frames in that cell), summarized by the median over the zone's cells -- independent of zone size and of time spent there; no area normalization needed."),
                    ("Lines", "one per zone x genotype combination -- color marks the genotype (same color in every zone), marker shape and line style mark the zone (see the separate Genotype and Zone legends)."),
                    ("How to read", "a rising or falling curve shows when during the session a genotype surveils a given zone more or less; compare colors within one line style for genotype differences in a zone, and line styles to each other for which part of the maze draws the most surveillance."),
                    ("Statistics", "none on this overlay; see Plot 2 for the zone-level pairwise comparison with significance testing."),
                ],
            },
            {
                "folder": "plot3b_temporal_median_velocity",
                "title": "3b Velocity over time",
                "summary": "Median velocity over time, one curve per zone, overlaid by genotype.",
                "items": [
                    ("X axis", "cumulative frames in the session (time bins)."),
                    ("Y axis", "for each session and time bin: the median, across the zone's grid cells, of each cell's mean velocity; points are the mean ± SE of those per-session medians."),
                    ("Normalization", "NOT normalized -- raw velocity (tracking units per second); a per-cell average summarized by the median over cells, so zone size doesn't inflate it."),
                    ("Lines", "one per zone x genotype combination -- color marks the genotype, marker shape and line style mark the zone."),
                    ("How to read", "a rising or falling curve shows when during the session a genotype's speed through a given zone changes."),
                    ("Statistics", "none on this overlay; see Plot 2b for the zone-level pairwise comparison with significance testing."),
                ],
            },
            {
                "folder": "plot4_surveillance_bout",
                "title": "4 Surveillance by bout type",
                "summary": "Surveillance probability at the top-corner decision zones, compared between successful and unsuccessful bouts, by genotype.",
                "items": [
                    ("X axis", "bout outcome (Successful, Unsuccessful -- whether the target zone was reached at least once during the bout)."),
                    ("Y axis", "per bout: fraction of the bout's frames in the zone that are Slow / Surveillance (state 1); median over bouts per session; bars are the mean ± SE across sessions."),
                    ("Normalization", "a ratio within the zone (state-1 frames ÷ the bout's frames in the zone), so it is independent of zone size and of time spent there; bouts that never enter the zone are left out."),
                    ("Bars", "one per genotype within each outcome (mean ± SE)."),
                    ("How to read", "a genotype surveilling more before an unsuccessful bout than a successful one may be hesitating or re-checking its route when it's about to get lost."),
                    ("Statistics", "within-genotype mixed model (Successful vs. Unsuccessful, session as a random effect) with FDR correction, and across-genotype t-tests within each outcome, both in the Statistics table below."),
                ],
            },
            {
                "folder": "plot4b_surveillance_bout_time_bins",
                "selectors": [{"label": "Time bins", "options": [("pct_bins", "% of session"), ("frame_bins", "Frames")]}],
                "title": "4b Surveillance by bout type over time",
                "summary": "Plot 4 split by when in the session the bout starts: surveillance probability at the decision zone across 5 time bins, successful and unsuccessful bouts side by side, by genotype.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Time bins", "chosen with the Time bins selector above. % of session = 5 equal parts of each session; Frames = frames 0-10k, 10k-20k, ..., 40k-50k (bouts starting later are left out). Each bout goes in the bin of its first frame (only ~3% of bouts cross a bin edge)."),
                    ("Panels", "Successful | Unsuccessful, on a shared y axis."),
                    ("X axis", "time bin."),
                    ("Y axis", "as Plot 4: per bout, the fraction of its frames in the zone that are Slow / Surveillance (state 1); per mouse, the median over its bouts in that bin; lines are the mean ± SE across mice."),
                    ("n labels", "\"n=X\" by a point = fewer mice than the group's n had such a bout in that bin (mostly the later Frames bins)."),
                    ("Normalization", "as Plot 4: a ratio within the zone, independent of zone size and time spent there; bouts that never enter the zone are left out."),
                    ("How to read", "lines that diverge over the session (rather than sitting apart throughout) mean the groups change differently as the session goes on -- e.g. with learning or fatigue."),
                    ("Statistics", [
                        "per outcome, a mixed model probability ~ genotype × time bin (session as a random effect), with likelihood-ratio tests for the genotype × time bin interaction (do groups change differently over the session?) and for genotype.",
                        "across-genotype Welch t-tests in each bin, FDR-corrected across the 5 bins.",
                    ]),
                ],
            },
            {
                "folder": "plot5_bout_depth",
                "title": "5 Bout depth & direction",
                "summary": "How far into the maze each bout goes before turning back, and how often mice move against their bout's direction (backtracking on the way in, re-entering on the way out).",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Path position", "the maze is one path with no branches, so every cell has a position along it: lane 0 runs from the entry (cells 0/1) to the far side, a U-turn leads into lane 1 running back, and so on, alternating sides -- the reward (70/71/80/81) is the U-turn after lane 7, and lanes 8-9 are the overshoot ending at the terminal (90/91). On this dataset 97% of cell-to-cell moves change the position by 1-2 steps, so the ordering matches how mice actually move."),
                    ("Bouts", "entry zone -> into the maze -> back to the entry zone; bouts under 20 frames and each session's last bout, if the session ended before the mouse came back, are left out."),
                    ("Direction (per frame)", "the sign of the net path change from 10 frames before to 10 frames after (±2 s at 5 fps), ignoring tracking jumps of more than 3 positions: forward (toward the reward/terminal) or backward (toward the entry). A frame with no net movement (a pause) takes the direction of the nearer moving frame in time -- a pause splits at its midpoint, the first half going with the movement before it, the second half with the movement after it -- so pauses count toward the stretches they interrupt, including their low speed."),
                    ("Phase (per frame)", [
                        "reward-reaching bouts (deepest point at the reward or beyond) turn at their FIRST ARRIVAL at the reward (cells 70/71/80/81): before it, forward = outbound, backward = backtrack; after it, by zone -- reward zone = at reward; overshoot (cells 82-99): forward (deeper) = overshoot forward, backward (toward the reward) = overshoot backward; path: backward (toward the entry) = return, forward (back toward the reward) = re-entry. These 7 phases cover every frame of the bout.",
                        "bouts that don't reach the reward turn at their furthest point -- the stay at the furthest path position, from first arrival, split at its midpoint (first half outbound, second half return): before the midpoint, forward = outbound, backward = backtrack; after it, backward = return, forward = re-entry.",
                        "data_with_phases.csv also has a 'moving' column (False = a pause that was assigned to its neighbours).",
                        "every frame's label is saved to results/compass_level_1/data_with_phases.csv (same rows as data_with_states.csv; frames between bouts are 'entry_zone').",
                    ]),
                    ("Left panel", "% of each session's bouts whose deepest point is in each lane (0 = entry lane, 7 = reward, highlighted; 9 = terminal); bars are the mean ± SE across sessions."),
                    ("Right panel", "per session: % of bouts reaching the reward; mean ± SE across sessions. Backtrack and re-entry are shown in Plot 5b, separately for bouts that reached the reward and bouts that did not -- the phases are defined differently for the two."),
                    ("Normalization", "by each session's own bout count, so sessions with more bouts are comparable; not by zone size."),
                    ("How to read", "a group skewed toward lane 9 explores to the far end more often."),
                    ("Statistics", "descriptive tables (mean (SE) across sessions) for depth by lane and for % of bouts reaching the reward, in the Statistics table below; no significance tests yet."),
                ],
            },
            {
                "folder": "plot5b_phase_fractions",
                "title": "5b Time by direction",
                "summary": "How each genotype group's bout time divides between Plot 5's phases -- going in, backtracking, at the reward, in the overshoot, coming back, re-entering -- separately for bouts that reached the reward and bouts that did not.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Figures", "two per comparison: bouts that reached the reward (deepest point at the reward or beyond) first, then bouts that did not."),
                    ("Bars", "per genotype group, one bar per phase (Plot 5, pauses included in the stretch they interrupt): bouts that reached the reward have 7 -- outbound, backtrack, at reward, overshoot forward, overshoot backward, return, re-entry; bouts that did not have 4 -- outbound, backtrack, return, re-entry."),
                    ("Normalization", "fractions of each session's own frames in that bout type, so the bars sum to 1 per session (and their means sum to 1); mean ± SE across sessions. Not by zone size."),
                    ("How to read", "a direct trip is mostly outbound + return (+ time at the reward and in the overshoot); more backtrack or re-entry means more reversals on the path. Splitting by bout type keeps short aborted bouts from diluting deep ones, which matters because deep bouts have more room to reverse."),
                    ("Statistics", "Welch t-tests between every pair of genotype groups for each phase, within each bout type, with Benjamini-Hochberg FDR across the phases, in the Statistics table below."),
                ],
            },
            {
                "folder": "plot5c_bout_length",
                "selectors": [{**TIME_BINS_SELECTOR, "none_label": "Whole session"}],
                "title": "5c Bout time",
                "summary": "How many frames a bout takes, for bouts that reached the reward and bouts that did not, and where a successful bout's time goes, split into Plot 5's phases with the reward zone split into the first and later visits.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Bouts", "entry zone -> into the maze -> back to the entry zone. Successful = reached the target zone (70/71/80/81) at least once, as in Plot 4. A session's first bout (if it began outside the entry zone) and last bout (if the session ended before the mouse came back) are left out, since they are cut off; no minimum length."),
                    ("Frames", "analyzed (tracked) frames, the unit used throughout; seconds on the second axis = frames ÷ 5 fps. A bout with tracking gaps counts fewer frames than the time it spanned (half of bouts are >= 97% tracked; 5% are under 56%)."),
                    ("Whole session", [
                        "top left: whole-bout length, Successful vs Unsuccessful -- dots = each mouse's median over its bouts, bars = mean ± SE across mice.",
                        "top right: the 8 parts of the successful bouts, which add up to every frame of the bout; same y axis as the left panel. They are Plot 5's phases for reward-reaching bouts -- outbound and backtrack (on the path before the first arrival at the reward), overshoot forward and backward (in the overshoot, cells 82-99), return and re-entry (on the path after the first arrival: heading home / heading back to the reward) -- with the reward zone split into the first visit (from the first arrival until it first leaves) and later visits (every later frame there). Each is all such frames of the bout added together. Parts a bout doesn't have (often backtrack, re-entry, the overshoot) are left out of its medians, since 0 can't be shown on a log axis.",
                        "bottom left: where the time goes -- the share of successful-bout frames in each part, stacked to 100% (per mouse: the part's frames over all its successful bouts ÷ their total frames; bars = the mean across mice, labels >= 5%).",
                        "bottom right: cumulative distribution of successful bouts' lengths per group, each mouse weighted equally (a mouse with many bouts doesn't dominate); a curve further right = longer bouts; a long right tail = some very long bouts.",
                    ]),
                    ("Time bins", "chosen with the Time bins selector above: each bout goes in the bin of its first frame (% of session = 5 equal parts; Frames = 0-10k ... 40k-50k, later bouts left out); four panels per row, in this order: Successful, Unsuccessful, Outbound, Return, Reward first visit, Reward later visits, Backtrack, Re-entry, Overshoot forward, Overshoot backward; lines = mean ± SE across mice of each mouse's median in that bin; \"n=X\" by a point = fewer mice than the group's n had such a bout there."),
                    ("Y axis", "log scale -- lengths are heavily skewed (successful bouts: median ~800 frames, a few over 10,000)."),
                    ("Normalization", "none -- raw frame counts; per-mouse medians so each mouse counts once and a few very long bouts don't dominate."),
                    ("Statistics", [
                        "Welch t-tests between genotype groups on log10 of the per-mouse medians, per panel, FDR across the table.",
                        "mixed models on every bout's log10 length (session as a random effect), likelihood-ratio tests: genotype × outcome (does the successful-vs-unsuccessful gap differ between groups?) and genotype; for each part of the successful bouts, genotype.",
                        "time bins: per panel, a mixed model log10 length ~ genotype × time bin with LRTs for the interaction and genotype, and per-bin t-tests with FDR across the 5 bins.",
                    ]),
                ],
            },
            {
                "folder": "plot5d_bout_distance",
                "selectors": [{**TIME_BINS_SELECTOR, "none_label": "Whole session"}],
                "title": "5d Bout distance",
                "summary": "5c with distance instead of time: how far a bout goes (cells moved along the path), for bouts that reached the reward and bouts that did not, and where a successful bout's distance goes, split into the same 8 parts.",
                "items": [
                    ("Distance", "cells moved along the path, frame to frame (1 cell = 1 step along the path; a lane is 10 cells); jumps of more than 3 positions are tracking errors and count 0, as in Plot 5. A frame's movement counts toward the part of the frame it arrives in. Unlike time, standing still adds nothing -- pauses take time but no distance. Everything else is as 5c, with distance in place of time and no seconds axis."),
                    ("Comparison", COMPARISON_BULLETS),
                    ("Bouts", "entry zone -> into the maze -> back to the entry zone. Successful = reached the target zone (70/71/80/81) at least once, as in Plot 4. A session's first bout (if it began outside the entry zone) and last bout (if the session ended before the mouse came back) are left out, since they are cut off; no minimum length."),
                    ("Frames", "analyzed (tracked) frames, the unit used throughout; seconds on the second axis = frames ÷ 5 fps. A bout with tracking gaps counts fewer frames than the time it spanned (half of bouts are >= 97% tracked; 5% are under 56%)."),
                    ("Whole session", [
                        "top left: whole-bout length, Successful vs Unsuccessful -- dots = each mouse's median over its bouts, bars = mean ± SE across mice.",
                        "top right: the 8 parts of the successful bouts, which add up to every frame of the bout; same y axis as the left panel. They are Plot 5's phases for reward-reaching bouts -- outbound and backtrack (on the path before the first arrival at the reward), overshoot forward and backward (in the overshoot, cells 82-99), return and re-entry (on the path after the first arrival: heading home / heading back to the reward) -- with the reward zone split into the first visit (from the first arrival until it first leaves) and later visits (every later frame there). Each is all such frames of the bout added together. Parts a bout doesn't have (often backtrack, re-entry, the overshoot) are left out of its medians, since 0 can't be shown on a log axis.",
                        "bottom left: where the time goes -- the share of successful-bout frames in each part, stacked to 100% (per mouse: the part's frames over all its successful bouts ÷ their total frames; bars = the mean across mice, labels >= 5%).",
                        "bottom right: cumulative distribution of successful bouts' lengths per group, each mouse weighted equally (a mouse with many bouts doesn't dominate); a curve further right = longer bouts; a long right tail = some very long bouts.",
                    ]),
                    ("Time bins", "chosen with the Time bins selector above: each bout goes in the bin of its first frame (% of session = 5 equal parts; Frames = 0-10k ... 40k-50k, later bouts left out); four panels per row, in this order: Successful, Unsuccessful, Outbound, Return, Reward first visit, Reward later visits, Backtrack, Re-entry, Overshoot forward, Overshoot backward; lines = mean ± SE across mice of each mouse's median in that bin; \"n=X\" by a point = fewer mice than the group's n had such a bout there."),
                    ("Y axis", "linear (cells moved) -- distances are much less skewed than times; the tests still use log10 of the per-mouse medians. The parts panel and each time-bin panel have their own y axis, so the small parts (backtrack, the reward visits) aren't flattened by the whole bouts -- check the axis values when comparing panels."),
                    ("Normalization", "none -- raw frame counts; per-mouse medians so each mouse counts once and a few very long bouts don't dominate."),
                    ("Statistics", [
                        "Welch t-tests between genotype groups on log10 of the per-mouse medians, per panel, FDR across the table.",
                        "mixed models on every bout's log10 length (session as a random effect), likelihood-ratio tests: genotype × outcome (does the successful-vs-unsuccessful gap differ between groups?) and genotype; for each part of the successful bouts, genotype.",
                        "time bins: per panel, a mixed model log10 length ~ genotype × time bin with LRTs for the interaction and genotype, and per-bin t-tests with FDR across the 5 bins.",
                    ]),
                ],
            },
            {
                "folder": "plot5e_reversal_distance",
                "selectors": [{**TIME_BINS_SELECTOR, "none_label": "Whole session"}],
                "title": "5e Backtrack & re-entry distance",
                "summary": "Plot 5's backtrack and re-entry rates broken down: how often a mouse reverses per bout, how far each reversal goes, and in which lane it turns around again.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Bouts", "all complete bouts, successful or not (as Plot 5)."),
                    ("Occurrence", "a run of frames moving the same way that starts as a backtrack or a re-entry; a re-entry that carries on into the reward and the overshoot lasts until the mouse turns again, so it can end up in lanes 8-9. In detail: one run of consecutive backtrack frames (backing up toward the entry before the bout's turning point -- the first arrival at the reward, or the furthest point if it never got there) or re-entry frames (heading back in on the path after it), phases as in Plot 5, pauses included. Both are on the path only; moving back and forth in the overshoot is overshoot forward / backward, not a reversal here. Only occurrences that move at least 1 cell count (direction is judged over ±2 s, so some runs never leave their cell: about a quarter of backtrack and a tenth of re-entry runs)."),
                    ("Cells moved", "path positions covered by the occurrence, from the cell it was in just before through the furthest cell reached (1 cell = 1 step along the path; a lane is 10 cells)."),
                    ("Layout", "backtrack (top two rows), then re-entry (bottom two rows): first a row of occurrences per bout, the mean cells moved and how far (distance classes), then a row of where it turned around and where it ended up."),
                    ("Occurrences per bout", "each mouse's occurrences ÷ its complete bouts."),
                    ("How far", "% of a mouse's occurrences covering 1, 2-4, 5-9 or 10+ cells (10 = a full lane). Shown as a distribution because distances are bimodal -- many 1-cell moves (often just crossing back over a cell edge) and a separate group of long reversals of a lane or more -- so a per-mouse median would flip between ~1 and ~10 depending on which group is slightly larger."),
                    ("Mean cells moved", "each mouse's mean over all its occurrences in the session (pulled up by the long reversals -- read it with the distance classes); each dot is one mouse."),
                    ("Where it turned around / ended up", "where each occurrence turned around and where it ended up, as % of a mouse's occurrences per lane (0 = entry lane, 7 = reward, highlighted; 8-9 = overshoot). Backtrack: turned around = the deepest point it backed up from; ended up = the point nearest the entry it backed up to. Re-entry: turned around = the point nearest the entry it headed in again from; ended up = the deepest point it went back in to. Mice with no such occurrence are left out of these panels. Lanes 8-9 stay empty except where a re-entry ended up: a backtrack happens on the path before the reward, and a re-entry starts on the path, so only its end can be in the reward or overshoot."),
                    ("Bars", "mean ± SE across mice; dots = mice."),
                    ("Time bins", "chosen with the Time bins selector above: each occurrence goes in the bin of its bout's first frame (% of session = 5 equal parts; Frames = 0-10k ... 40k-50k, later bouts left out). Per reversal type, five panels by bin, laid out as the whole-session figure: occurrences per bout (that bin's occurrences ÷ its bouts), the mean number of cells an occurrence moves (each mouse's mean in that bin; 10 = one lane), % of occurrences going 10+ cells (a lane or more), then the mean lane where it turned around and where it ended up (reward lane 7 shaded); lines = mean ± SE across mice; \"n=X\" by a point = fewer mice than the group's n had a bout (left) or an occurrence (middle, right) in that bin."),
                    ("Normalization", "per bout (left); % of each mouse's own occurrences (middle and right), so each panel's bars sum to 100 per mouse; not by lane size."),
                    ("Statistics", "Welch t-tests between every pair of genotype groups for occurrences per bout, mean cells moved, % in each distance class, % turning around in and % ending up in each lane, per reversal type, with Benjamini-Hochberg FDR across those 26 tests per pair. Time bins: per reversal type and measure, a mixed model measure ~ genotype × time bin (session as a random effect) with likelihood-ratio tests for the interaction and genotype, and per-bin Welch t-tests with FDR across the 5 bins."),
                ],
            },
            {
                "folder": "plot5f_turn_heatmap",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [TIME_BINS_SELECTOR],
                "title": "5f Where mice turn around",
                "summary": "Cell by cell, how often a pass through the cell turns around there -- turning back while going deeper (backtrack) and turning in again while coming back (re-entry) -- on the path, at the reward and in the overshoot.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "to the right of each row: second group minus first for that probability, cell by cell, on a blue-white-red scale centered at 0 (red = more turns in the second group, blue = fewer, white = about the same). Two groups: B \u2212 A. Four groups: E3NLF \u2212 E3, E4NLF \u2212 E4, E4 \u2212 E3, E4NLF \u2212 E3NLF. Both rows share one color limit (95th percentile of their differences)."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Pass", "one stay in a cell, from entering to leaving it; it enters with the direction of travel of the frame before (Plot 5's direction, judged over ±2 s, pauses joined to the movement they interrupt)."),
                    ("Rows", [
                        "Backtrack (turn back): of the passes entering while going deeper -- outbound on the path, in the reward zone, overshoot forward -- the share that switch to moving back (backtrack, reward zone moving back, overshoot backward) in the cell.",
                        "Re-entry (turn in again): of the passes entering while coming back -- return on the path, in the reward zone, overshoot backward -- the share that switch to moving deeper (re-entry, reward zone moving deeper, overshoot forward) in the cell.",
                        "on the path, a backtrack happens only before the reward is reached and a re-entry only after; in the reward zone and the overshoot, both can happen at any time after the first arrival. The furthest point of a bout that never reaches the reward is its turnaround (outbound -> return), not a backtrack.",
                    ]),
                    ("Color", "per mouse, then averaged across the group's mice; 0 = never turns there; both rows on the same 0-0.2 orange scale (the terminal cells, where mice have to turn, go above it -- their numbers are shown)."),
                    ("Grey cells", "fewer than 5 such passes for a mouse in that cell (3 per time bin)."),
                    ("How to read", "on straight lanes nearly every pass carries on (about 0-1% turn); turns cluster at the U-turn corners, at the reward and at the end of the overshoot."),
                    ("Normalization", "a share of the mouse's own passes through that cell, so neither cell size nor time spent there inflates it; each mouse counts once."),
                    ("Statistics", "none on the maps."),
                ],
            },
            {
                "folder": "plot6a_direction_heatmap_surveillance",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "zone_set_none": True,  # "none" = the per-cell maps; zone sets = pooled by zone, one direction each
                "selectors": [DIRECTION_SELECTOR, TIME_BINS_SELECTOR],
                "title": "6a Surveillance heatmap by direction",
                "summary": "Plot 1's Slow / Surveillance maze heatmap split by direction of travel -- going in, coming back, backtrack, re-entry, or coming back minus going in -- pick one with the Direction selector.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "to the right of each row: second group minus first for that row, cell by cell, on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same). Two groups: B \u2212 A. Four groups: E3NLF \u2212 E3, E4NLF \u2212 E4, E4 \u2212 E3, E4NLF \u2212 E3NLF. Color limit: the 95th percentile of the differences; the Coming back \u2212 going in map has no group-difference panels."),
                    ("Time bins", TIME_BINS_ITEM),
                    TRAVEL_ITEM,
                    ("Panels", "one per genotype group, then the group differences; Coming back \u2212 going in = each mouse's own difference, averaged. The reward is outlined in gold."),
                    ("Zone sets", [
                        "none = the per-cell maps. A zone set pools each zone's frames in that direction (per mouse, >= 20 frames; then the mean across mice); cells outside the set are hatched.",
                        "outbound_entry_exit (with Going in) and return_entry_exit (with Coming back): the cells just before (entry) and just after (exit) the U-turns, bottom (cells x7-x9, 4 per zone) and top (cells x0-x2, 3 per zone) as separate zones, 2 / 1 / 0 = two cells / one cell / next to the wall (12 zones). Each goes only with its own direction -- its pill is greyed out for the others.",
                        "statistics: Welch t-tests between genotype groups per zone, Benjamini-Hochberg FDR across the set's zones, in the Statistics table below.",
                        "difference panels: second group minus first, per zone, as on the per-cell maps. Time bins: the zone-set versions are by time bin too (per mouse >= 10 frames per zone and bin); each zone's value is painted on its cells, cells outside the set grey.",
                    ]),
                    ("Color", "Slow / Surveillance (HMM state 1) share of that cell's frames in that direction, per mouse, then averaged across the group's mice; Coming back \u2212 going in is on a two-sided scale (red = more surveillance coming back, blue = less)."),
                    ("Normalization", "a ratio within each cell and phase, so neither cell size nor time spent there inflates it; each mouse counts once."),
                    ("Grey cells", "fewer than 20 frames for a mouse in that cell and phase (for the difference: in either phase) -- not enough to estimate."),
                    ("Statistics", "none on the maps."),
                ],
            },
            {
                "folder": "plot6b_direction_heatmap_velocity",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "zone_set_none": True,  # "none" = the per-cell maps; zone sets = pooled by zone, one direction each
                "selectors": [DIRECTION_SELECTOR, TIME_BINS_SELECTOR],
                "title": "6b Belly speed by direction",
                "summary": "Plot 1b's belly-speed maze heatmap split by direction of travel -- going in, coming back, backtrack, re-entry, or coming back minus going in -- pick one with the Direction selector.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "to the right of each row: second group minus first for that row, cell by cell, on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same). Two groups: B \u2212 A. Four groups: E3NLF \u2212 E3, E4NLF \u2212 E4, E4 \u2212 E3, E4NLF \u2212 E3NLF. Color limit: the 95th percentile of the differences; the Coming back \u2212 going in map has no group-difference panels."),
                    ("Time bins", TIME_BINS_ITEM),
                    TRAVEL_ITEM,
                    ("Panels", "one per genotype group, then the group differences; Coming back \u2212 going in = each mouse's own difference, averaged."),
                    ("Zone sets", [
                        "none = the per-cell maps. A zone set pools each zone's frames in that direction (per mouse, >= 20 frames; then the mean across mice); cells outside the set are hatched.",
                        "outbound_entry_exit (with Going in) and return_entry_exit (with Coming back): the cells just before (entry) and just after (exit) the U-turns, bottom (cells x7-x9, 4 per zone) and top (cells x0-x2, 3 per zone) as separate zones, 2 / 1 / 0 = two cells / one cell / next to the wall (12 zones). Each goes only with its own direction -- its pill is greyed out for the others.",
                        "statistics: Welch t-tests between genotype groups per zone, Benjamini-Hochberg FDR across the set's zones, in the Statistics table below.",
                        "difference panels: second group minus first, per zone, as on the per-cell maps. Time bins: the zone-set versions are by time bin too (per mouse >= 10 frames per zone and bin); each zone's value is painted on its cells, cells outside the set grey.",
                    ]),
                    ("Color", "mean belly speed (the point the whole pipeline tracks) of that cell's frames in that direction (tracking units per second), per mouse, then averaged; Coming back \u2212 going in: red = faster coming back."),
                    ("Normalization", "NOT normalized -- raw speed (distance per frame x 5 fps, no sign: moving forward or backward at the same pace gives the same value); an average, so cell size doesn't inflate it."),
                    ("Grey cells", "fewer than 20 frames for a mouse in that cell and phase."),
                    ("Statistics", "none on the maps."),
                ],
            },
            {
                "folder": "plot6c_temporal_direction_surveillance",
                "selectors": [DIRECTION_SELECTOR],
                "title": "6c Surveillance over time by direction",
                "summary": "Plot 3 for the direction zone sets: median Surveillance probability over the session per zone, counting only the frames moving in the zone set's direction.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Direction", "chosen with the Direction selector above; each zone set goes with one direction -- the *_outbound_* sets with Going in, the *_return_* sets (the same cells, entry and exit swapped) with Coming back -- so the other zone sets are greyed out. Going in = outbound + reward zone and overshoot moving deeper; coming back = return + reward zone and overshoot moving back (Plot 5's phases)."),
                    ("Zone sets", "8 sets, bottom / top x outbound / return x entry / exit, 3 zones each (2 / 1 / 0 = two cells / one cell / next to the wall): bottom = the lanes' cells x7-x9 next to the bottom U-turns (4 cells per zone), top = cells x0-x2 next to the top U-turns (3 per zone); entry = just before a U-turn in that direction, exit = just after."),
                    ("X axis", "cumulative frames in the session (2000-frame bins, as Plot 3); only bins every session reaches."),
                    ("Y axis", "for each session and time bin: the median, across the zone's cells, of each cell's Slow / Surveillance share over its frames moving in that direction; points are the mean ± SE of those per-session medians."),
                    ("Lines", "one per zone x genotype -- color marks the genotype, marker shape and line style the zone."),
                    ("Normalization", "a ratio within each cell, summarized by the median over the zone's cells -- independent of zone size and time spent there."),
                    ("Statistics", "none on this overlay; see 6a's zone sets for per-zone tests."),
                ],
            },
            {
                "folder": "plot6d_temporal_direction_velocity",
                "selectors": [DIRECTION_SELECTOR],
                "title": "6d Velocity over time by direction",
                "summary": "Plot 3b for the direction zone sets: median velocity over the session per zone, counting only the frames moving in the zone set's direction.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Direction", "chosen with the Direction selector above; each zone set goes with one direction -- the *_outbound_* sets with Going in, the *_return_* sets (the same cells, entry and exit swapped) with Coming back -- so the other zone sets are greyed out. Going in = outbound + reward zone and overshoot moving deeper; coming back = return + reward zone and overshoot moving back (Plot 5's phases)."),
                    ("Zone sets", "8 sets, bottom / top x outbound / return x entry / exit, 3 zones each (2 / 1 / 0 = two cells / one cell / next to the wall): bottom = the lanes' cells x7-x9 next to the bottom U-turns (4 cells per zone), top = cells x0-x2 next to the top U-turns (3 per zone); entry = just before a U-turn in that direction, exit = just after."),
                    ("X axis", "cumulative frames in the session (2000-frame bins, as Plot 3); only bins every session reaches."),
                    ("Y axis", "for each session and time bin: the median, across the zone's cells, of each cell's mean belly speed over its frames moving in that direction; points are the mean ± SE of those per-session medians."),
                    ("Lines", "one per zone x genotype -- color marks the genotype, marker shape and line style the zone."),
                    ("Normalization", "NOT normalized -- raw speed; a per-cell average summarized by the median over cells."),
                    ("Statistics", "none on this overlay; see 6b's zone sets for per-zone tests."),
                ],
            },
            {
                "folder": "plot7_head_angle_by_direction",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [DIRECTION_SELECTOR, TIME_BINS_SELECTOR],
                "title": "7 Head angle by direction",
                "summary": "Which side the head points to relative to the body, cell by cell, for each direction of travel (going in, coming back, backtrack, re-entry) -- pick one with the Direction selector.",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "second group minus first, cell by cell (or zone by zone), on a blue-white-red scale centered at 0 (red = higher in the second group, blue = lower, white = about the same), sized to the 95th percentile of the differences. No arrows on the difference panels. Two groups: B \u2212 A underneath. Four groups (second row): the NLF effect within each APOE genotype (E3NLF \u2212 E3, E4NLF \u2212 E4) and the APOE4 effect within WT and NLF (E4 \u2212 E3, E4NLF \u2212 E3NLF)."),
                    ("Time bins", TIME_BINS_ITEM),
                    TRAVEL_ITEM,
                    ("Why separate", "the directions are kept apart because the mouse's right points to opposite sides of the maze when it travels the other way, so pooling them would cancel the lean."),
                    ("Head angle", "the direction of the nose from the sternum relative to the body axis (tail base -> sternum), in degrees: 0 = straight ahead, + = the mouse's right, - = its left (assuming an unmirrored top-down camera)."),
                    ("Map", "the whole maze, one panel per genotype group; each cell is colored by the mean signed head angle (red = right, blue = left, white = none), with its value in the cell's bottom-right corner; the reward is outlined in gold."),
                    ("Arrows", [
                        "grey arrow = the body (tail base -> sternum; the arrowhead is the sternum, at the cell center).",
                        "black arrow = the head (sternum -> nose; the arrowhead is the nose).",
                        "calculated from the cell's mean signed head angle: the body arrow is drawn along the maze path for that direction, and the head arrow is the body arrow turned by the angle (right for +, left for -).",
                    ]),
                    ("How it's calculated", "per mouse, the mean signed head angle over its frames in that cell and phase (needs ≥ 20 frames), then averaged across the group's mice; grey = no mouse had enough frames."),
                    ("Coverage", "going in and coming back: every cell, all mice. Backtrack and re-entry are on the path only and sparse -- many cells rest on few mice, so read them as rougher maps."),
                    ("How to read", "at the U-turns the head leans into the turn being taken -- so a corner is one color going in and the other coming back."),
                    ("Normalization", "NOT normalized -- an angle in degrees, averaged; each mouse counts once."),
                    ("Statistics", "none on the maps."),
                ],
            },
            {
                "folder": "plot7b_head_angle_abs_by_direction",
                "grid_sizing": True,  # maze heatmaps fit the card's width
                "selectors": [DIRECTION_SELECTOR, TIME_BINS_SELECTOR],
                "title": "7b Head angle lean (absolute) by direction",
                "summary": "Plot 7 with the sign dropped: each cell colored by the absolute value of its mean signed head angle -- how strongly the head leans, ignoring the side -- for each direction of travel (pick one with the Direction selector).",
                "items": [
                    ("Comparison", COMPARISON_BULLETS),
                    ("Difference panels", "|second group| minus |first group| of the cell's signed mean -- red = the head leans more strongly in the second group, blue = less, white = about the same -- on a blue-white-red scale centered at 0. Two groups: B \u2212 A underneath; four groups: E3NLF \u2212 E3, E4NLF \u2212 E4, E4 \u2212 E3, E4NLF \u2212 E3NLF in a second row. No arrows on the difference panels."),
                    ("Time bins", TIME_BINS_ITEM),
                    ("Color", "the absolute value of Plot 7's number: |mean signed head angle| (e.g. -12 in Plot 7 is 12 here), with the value in the cell's bottom-right corner; same 0-30 deg range as Plot 7."),
                    ("Arrows", "as Plot 7: grey arrow = the body (tail base -> sternum), black arrow = the head (sternum -> nose), turned by the signed angle -- so the arrows show which side the head leans, while the color shows how strongly."),
                    ("Grey cells", "fewer than 20 frames with a head angle for a mouse in that cell and phase."),
                    ("Normalization", "NOT normalized -- an angle in degrees; each mouse counts once."),
                    ("Statistics", "none on the maps."),
                ],
            },
        ],
    },
}
