"""Prism-ready tables for the figure report: for each figure, the per-session
values behind it laid out for pasting into a GraphPad Prism data table.

Built from the datasheet workbooks the notebooks write (source_data.py) --
nothing here changes the analysis. build.py attaches each step's tables to
its cards, where the report offers them as a CSV download or a clipboard
copy, with a sticky comparison/zone selector choosing which one shows.

Layouts:
  grouped  rows (zones, bout types, ...) x one data set per genotype group,
           replicates = sessions in the same order on every row -- Prism's
           "Separated" box plot draws each group's boxes side by side.
  xy       X = time/bin, one data set per genotype group, replicates =
           sessions.
Each table is {"rows", "paste", "titles", "steps"} -- see _table().

This is a project-specific adapter (genotype groups come from config.yaml's
genotype_sets, not auto-detected) -- a future project with different zones/
genotypes only needs its own STEP_TABLES mapping below, not changes to the
generic _table()/grouped()/xy() layout builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- generic layout primitives (ported from the reference postanalysis report) ---

def _num(v, digits=10):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else float(f"{v:.{digits}g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _session_key(s):
    return (0, int(s)) if str(s).isdigit() else (1, str(s))


def _col(n):
    """1-based column number -> spreadsheet letters (1 -> A, 27 -> AA)."""
    letters = ""
    while n:
        n, r = divmod(n - 1, 26)
        letters = chr(65 + r) + letters
    return letters


def _letters(names):
    return ", ".join(f"{_col(i + 1)} = {n}" for i, n in enumerate(names))


REPLICATES = "**'Enter {n} replicate values in side-by-side subcolumns'**"
SHORT_GROUPS = "Groups with fewer sessions than {n} leave blank cells at the end of their block; that is expected."
PLACEHOLDER = "DELETE THIS ROW"


def _grouped_table(options):
    return ["Prism 11 > New Data Table > **Grouped**.", f"Options: {options}."]


def _xy_table(y):
    return ["Prism 11 > New Data Table > **XY**.", f"X: **numbers**; Y: {y}."]


def _table(rows, paste, set_titles, row_titles=None, *, table, what, graph, analysis, row_kind="", x_title=None,
           layout="", n_max=None, subcolumns=None, placeholder_row=None):
    """A Prism table: its rows (the CSV); `paste`, the value rows ('Copy
    table' copies rows[0] plus those rows); `subcolumns`, each data set's
    subcolumn titles [(data set, [title, ...]), ...] (the card's per-data-set
    'IDs' button); and the card's text (what/layout/steps/note)."""
    shown = _letters(set_titles) if len(set_titles) <= 12 else (
        f"{_letters(set_titles[:3])}, ... ({len(set_titles)} titles)")
    fills = (f"the X title ({x_title}), the data set titles ({shown}), and the X and Y values" if x_title is not None
             else f"the data set titles ({shown}), the row title ({row_titles[0]}) and the values" if len(row_titles) == 1
             else f"the data set titles ({shown}), the {len(row_titles)} row titles ({row_kind}) and the values")
    items = lambda x: [x] if isinstance(x, str) else list(x)
    steps = [["Table", items(table)],
             ["Data", ["Click **'Copy table'** on this card.",
                       "In Prism, click the **corner cell with the small gray triangle** -- the top-left cell of "
                       "the table, just above the row numbers 1, 2, 3 (it selects the whole table).",
                       f"Paste -- this fills {fills}."]
              + ([f"**Delete row {placeholder_row}** ('{PLACEHOLDER}'): click its row number to select it and "
                  f"delete the row -- it is only there because Prism keeps the titles of a pasted table only when "
                  f"it has 2+ rows."] if placeholder_row else [])],
             ["Graph", items(graph)],
             ["Analysis", items(analysis)]]
    if subcolumns:
        header = "A:Y1" if x_title is not None else "A:1"
        example = subcolumns[0][1][0] if subcolumns[0][1] else ""
        steps.insert(2, ["Mouse IDs (optional, to see which session each subcolumn is)", [
            f"Open: double-click the **{header}** subcolumn header (or Change > Format Data Table > subcolumn tab).",
            "**Select** 'Label the data table with these names'; **unselect** 'Enter one set of subcolumn titles'.",
            "Click **'Copy session IDs'**, click data set A's **first subcolumn title box**, and paste.",
            f"Result: every data set fills in order (A's first subcolumn: {example}); groups with fewer sessions "
            f"get blank titles at the end."]])
    r1 = paste["r"] + 1
    r2 = len(rows) if paste["n"] is None else paste["r"] + paste["n"]
    note = (f"Instead of pasting: File > Import the CSV download and import only rows {r1}-{r2}, columns "
            f"{_col(paste['c'] + 1)}-{_col(len(rows[0]))}.")
    if n_max is not None:
        note += " " + SHORT_GROUPS.format(n=n_max)
    return {"rows": rows, "paste": paste,
            "steps": {"about": what, "layout": layout, "steps": steps, "note": note},
            "titles": {"sets": [str(t) for t in set_titles],
                       "rows": [str(t) for t in row_titles] if row_titles is not None else None},
            "subcolumns": subcolumns}


def _n_label(design, group):
    return f"{group} (n={len(design.sessions.get(group, []))})"


def _header(design, groups, first=""):
    names, sessions = [first], ["session"]
    for g in groups:
        ss = design.sessions.get(g, []) + [""] * (design.n_max - len(design.sessions.get(g, [])))
        names += [_n_label(design, g)] + [""] * (design.n_max - 1)
        sessions += ss
    return [names, sessions]


def _subcolumn_titles(design, groups):
    return [(g, [f"s{s}" for s in design.sessions.get(g, [])]) for g in groups]


def _group_values(design, groups, series):
    row = []
    for g in groups:
        vals = [_num(series.get(s, np.nan)) for s in design.sessions.get(g, [])]
        row += vals + [""] * (design.n_max - len(vals))
    return row


def grouped(design, groups, row_labels, value_of, what, graph, analysis, row_kind="row labels", layout=""):
    """Rows = `row_labels` [(key, title), ...], one data set per group,
    replicates = sessions in the same order on every row. `value_of(key)` ->
    Series indexed by session."""
    rows = _header(design, groups)
    rows += [[label] + _group_values(design, groups, value_of(key)) for key, label in row_labels]
    single = len(row_labels) == 1
    if single:
        rows.append([PLACEHOLDER] + [""] * (len(rows[0]) - 1))
    return _table(rows, {"r": 2, "n": None, "c": 1}, [_n_label(design, g) for g in groups],
                  [label for _, label in row_labels],
                  table=_grouped_table(REPLICATES.format(n=design.n_max)), row_kind=row_kind, what=what,
                  graph=graph, analysis=analysis, n_max=design.n_max, subcolumns=_subcolumn_titles(design, groups),
                  layout=layout, placeholder_row=1 if single else None)


def xy(design, groups, x_values, value_of, x_title, what, graph, analysis):
    """X = `x_values`, one data set per group, replicates = sessions.
    `value_of(x)` -> Series indexed by session for that X."""
    rows = _header(design, groups, first=x_title)
    rows += [[_num(x)] + _group_values(design, groups, value_of(x)) for x in x_values]
    return _table(rows, {"r": 2, "n": None, "c": 0}, [_n_label(design, g) for g in groups],
                  table=_xy_table(REPLICATES.format(n=design.n_max)),
                  x_title=x_title, what=what, graph=graph,
                  analysis=analysis, n_max=design.n_max, subcolumns=_subcolumn_titles(design, groups),
                  layout="No row titles needed: the X values label the rows. Each subcolumn is the same session in "
                         "every row.")


# --- design (this project's genotype groups) -----------------------------------

class Design:
    """A genotype comparison's groups and their sessions, built from a
    datasheet's "mice" sheet (session, group=raw genotype) + a
    config.yaml genotype_sets[set_name] entry -- the Prism-layout
    equivalent of the report's own "Comparison" pills, reshaped for pasting
    into Prism instead of plotting."""

    def __init__(self, mice: pd.DataFrame, genotype_mapping: dict):
        mice = mice.copy()
        mice["session"] = mice["session"].astype(str)
        genotype_lookup = {raw: label for label, raws in genotype_mapping.items() for raw in raws}
        mice = mice[mice["group"].isin(genotype_lookup)].copy()
        mice["group"] = mice["group"].map(genotype_lookup)
        self.groups = list(genotype_mapping.keys())
        self.sessions = {
            g: sorted(mice.loc[mice["group"] == g, "session"], key=_session_key) for g in self.groups
        }
        self.n_max = max((len(s) for s in self.sessions.values()), default=0)


FDR_BIN_NOTE = ["Pipeline: FDR-corrected pairwise t-tests between genotypes, run separately per bin -- see the "
                "Statistics table below the figure.",
                "In Prism: Analyze > **Multiple t tests** (one unpaired t test per row), with a **false discovery "
                "rate (Benjamini-Hochberg) correction** -- same comparison as the pipeline's."]
FDR_ZONE_NOTE = ["Pipeline: FDR-corrected pairwise t-tests between genotypes, run separately per zone -- see the "
                 "Statistics table below the figure.",
                 "In Prism: Analyze > **Multiple t tests** (one unpaired t test per row), with a **false discovery "
                 "rate (Benjamini-Hochberg) correction** -- same comparison as the pipeline's."]
TTEST_NOTE = ["Pipeline: Welch's t-test (unequal variance) between genotypes -- see the figure and Statistics table.",
              "In Prism: Analyze > **t tests** (unpaired, Welch's correction)."]
G_BAR = ["Prism 11 > Graphs > Grouped > **Bar** (with individual points), or **Box plot**."]
G_POINTS = ["Prism 11 > Graphs > XY > **Points**, with error bars (mean, SE)."]


# --- comparison / zone-set tags read off a file name ------------------------------

def detect_key(stem: str, keys) -> str | None:
    """The first of `keys` appearing in `stem` as whole "_"-separated tokens
    (so "4_corners" matches "..._4_corners.pdf" but not "..._top_corners"),
    trying keys with more tokens first. A leading "all_genotypes_" (a file
    naming convention, not the "all" comparison) is ignored."""
    if stem.startswith("all_genotypes_"):
        stem = stem[len("all_genotypes_"):]
    tokens = stem.split("_")
    for key in sorted(keys, key=lambda k: len(k.split("_")), reverse=True):
        kt = key.split("_")
        n = len(kt)
        if any(tokens[i:i + n] == kt for i in range(len(tokens) - n + 1)):
            return key
    return None


def zone_set_of(stem: str, config: dict) -> str | None:
    """The zone set a figure/table was made with, from its name: a
    config.yaml zone_sets key (e.g. "9_corners"), else -- for a step built
    on one zone rather than a set (e.g. "top_corners") -- a zones key."""
    return detect_key(stem, list(config.get("zone_sets") or {})) or detect_key(stem, list(config.get("zones") or {}))


# --- helpers shared by the per-step builders below ------------------------------

_NON_ZONE_COLS = {"session", "mouse_id", "group", "sex", "timepoint", "genotype", "hmm_state", "cnt", "tot",
                  "figure", "successful_only", "pct_of_bouts", "_zone_set"}


def _infer_zone_col(df: pd.DataFrame, value_col: str) -> str | None:
    """The one categorical column left in a recorded sheet once the known
    metadata/value columns are excluded -- the zone/node-type/outcome a
    row's value belongs to, whatever that column happens to be named for
    this particular step. `df` should already be filtered to this step's
    own rows (see build_prism_store()'s dropna on value_col); a column
    that's entirely NaN here belongs to a different metric sharing the same
    sheet (e.g. "velocity" on plot2's rows of a sheet shared with plot2b)
    and is excluded too, not just the fixed metadata/value names."""
    candidates = [c for c in df.columns if c not in _NON_ZONE_COLS and c != value_col and df[c].notna().any()]
    return candidates[0] if candidates else None


def _series_of(df: pd.DataFrame, col: str, value_col: str) -> dict:
    """{column value: Series(session -> value_col)} -- one group per value
    of `col`, in first-seen order."""
    out = {}
    for key in dict.fromkeys(df[col]):
        out[key] = _per_session(df.loc[df[col] == key], value_col)
    return out


def _per_session(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Series(session -> value_col), one value per session. A shared sheet
    can hold the same session more than once (the same metric drawn by
    several figures, e.g. under different genotype comparisons -- the
    per-session value itself doesn't depend on how genotypes are pooled),
    and a duplicated index would break the per-session lookups."""
    return df.groupby("session")[value_col].first()


def _grouped_single(sheet, design, *, value_col, label, what, graph, analysis):
    """One value per session, grouped by genotype -- a single-row Prism table."""
    values = _per_session(sheet, value_col)
    t = grouped(design, design.groups, [(value_col, label)], {value_col: values}.get, what, graph, analysis,
                layout=f"One row ({label}), one data set per genotype group.")
    return [(None, t)]


def _grouped_rows(sheet, design, *, zone_col, value_col, what, graph, analysis, row_kind):
    """Rows = every value of `zone_col` (e.g. zone, bout outcome), one data
    set per genotype group -- all of them in ONE table, no zone selector
    needed (matches the figure, which already shows every zone as its own
    row/box group)."""
    series_by = _series_of(sheet, zone_col, value_col)
    row_labels = [(k, str(k)) for k in series_by]
    t = grouped(design, design.groups, row_labels, series_by.get, what, graph, analysis, row_kind=row_kind,
                layout=f"rows = {row_kind}, one data set per genotype group, matching the figure.")
    return [(None, t)]


def _xy(sheet, design, *, x_col, value_col, x_title, what, graph, analysis):
    """X = every value of `x_col` (e.g. time bin), one data set per genotype
    group, no zone selector."""
    series_by_x = _series_of(sheet, x_col, value_col)
    x_values = sorted(series_by_x, key=lambda v: (isinstance(v, str), v))
    t = xy(design, design.groups, x_values, series_by_x.get, x_title, what, graph, analysis)
    return [(None, t)]


def xy_all_zones(design, zones, x_values, value_of, x_title, what, graph, analysis):
    """X = `x_values`, one data set per genotype group x zone (zone-major, as
    in the figure's legend), replicates = sessions. `value_of(zone, x)` ->
    Series indexed by session."""
    sets = [(g, z) for z in zones for g in design.groups]
    names, sessions = [x_title], ["session"]
    for g, z in sets:
        ss = design.sessions.get(g, [])
        names += [f"{_n_label(design, g)} · {z}"] + [""] * (design.n_max - 1)
        sessions += ss + [""] * (design.n_max - len(ss))
    rows = [names, sessions]
    for x in x_values:
        row = [_num(x)]
        for g, z in sets:
            series = value_of(z, x)
            vals = [_num(series.get(s, np.nan)) for s in design.sessions.get(g, [])]
            row += vals + [""] * (design.n_max - len(vals))
        rows.append(row)
    titles = [f"{_n_label(design, g)} · {z}" for g, z in sets]
    return _table(rows, {"r": 2, "n": None, "c": 0}, titles,
                  table=_xy_table(REPLICATES.format(n=design.n_max)),
                  x_title=x_title, what=what, graph=graph, analysis=analysis, n_max=design.n_max,
                  subcolumns=[(t, [f"s{s}" for s in design.sessions.get(g, [])]) for t, (g, _) in zip(titles, sets)],
                  layout="Every zone in one table, like the figure: one data set per genotype group x zone. In the "
                         "graph, color data sets by genotype and give each zone its own symbol / line style.")


def _xy_by_zone(sheet, design, *, zone_col, x_col, value_col, x_title, what, graph, analysis):
    """One XY table with every value of `zone_col` (zone/region) in it, as
    the figure draws them: X = x_col, one data set per genotype group x
    zone -- no zone selector."""
    zones = list(dict.fromkeys(sheet[zone_col]))
    by_zone = {z: _series_of(sheet.loc[sheet[zone_col] == z], x_col, value_col) for z in zones}
    x_values = sorted({x for s in by_zone.values() for x in s}, key=lambda v: (isinstance(v, str), v))
    empty = pd.Series(dtype=float)
    t = xy_all_zones(design, zones, x_values, lambda z, x: by_zone[z].get(x, empty),
                     x_title, f"{what} (all zones).", graph, analysis)
    return [(None, t)]


# --- per-step dispatch -----------------------------------------------------------
# {pipeline: {step folder: (sheet name, builder, kwargs) | None}} -- None
# means the step's recorded data is already aggregated across sessions (a
# spatial grid heatmap) and so has no per-session replicates left to paste
# into Prism. The sheet name is the source_data.flush() name the step's own
# plotting function uses (see each file's record()/flush() call) -- several
# steps sharing the exact same sheet (e.g. a step and its "one page per
# zone" sibling, or Plot 1/1b sharing "grid_cells") is expected, not a bug:
# the datasheet is organized by what was computed, not one sheet per figure.

STEP_SPECS: dict[str, dict[str, tuple | None]] = {
    "task_performance": {
        "exclusion_criteria": ("sessions", _grouped_single, dict(
            value_col="target_usage", label="Target zone usage",
            what="Fraction of the session spent in the target zone, one value per session.",
            graph=G_BAR, analysis=TTEST_NOTE)),
        "plot1_heatmap": None,
        "plot1b_heatmap_pct": None,
        "plot2_entropy": ("sessions_x_bins", _xy, dict(
            x_col="bin", value_col="entropy", x_title="Cumulative frames",
            what="Shannon entropy of region occupancy, one value per session per bin.",
            graph=G_POINTS, analysis=FDR_BIN_NOTE)),
        "plot3_all_regions_usage": ("sessions_x_bins_x_regions", _xy_by_zone, dict(
            zone_col="region", x_col="bin", value_col="proportion", x_title="Cumulative frames",
            what="Proportion of that bin's time spent in the region, one value per session per bin.",
            graph=G_POINTS, analysis=FDR_ZONE_NOTE)),
        "plot3b_all_regions_usage_pct": None,
        "plot4_success_rate": ("sessions", _grouped_single, dict(
            value_col="success_rate", label="% successful bouts",
            what="% of bouts, over the whole session, that reached the target zone -- one value per session.",
            graph=G_BAR, analysis=TTEST_NOTE)),
        "plot5_time_based_success": ("sessions_x_bins", _xy, dict(
            x_col="bout_num", value_col="succ_bout_perc", x_title="Cumulative frames",
            what="% of bouts started in that bin that reached the target zone, one value per session per bin.",
            graph=G_POINTS, analysis=FDR_BIN_NOTE)),
        "plot5b_time_based_success_pct": None,
        "plot6_zone_occupancy_velocity": ("sessions_x_bouts_x_zones", _xy_by_zone, dict(
            zone_col="zone", x_col="ind_no", value_col="occupancy", x_title="Bout number (binned)",
            what="Fraction of that bout spent in the zone, one value per session per bout-bin.",
            graph=G_POINTS, analysis=["none in the pipeline (a descriptive, exponential-fit overlay)."],
            where={"pct_of_bouts": False, "successful_only": False})),
        "plot6b_zone_occupancy_velocity_pct": None,
    },
    "level_1": {
        "plot1_heatmap": None,
        "plot1b_velocity_heatmap": None,
        "plot2_state_probability": ("sessions_x_zones", _grouped_rows, dict(
            zone_col=None, value_col="prop", row_kind="hairpin-turn zones",
            what="Probability of the Slow / Surveillance state (HMM state 1) in that zone, one value per session.",
            graph=G_BAR, analysis=FDR_ZONE_NOTE)),
        "plot2b_velocity": ("sessions_x_zones", _grouped_rows, dict(
            zone_col=None, value_col="velocity", row_kind="column zones",
            what="Mean velocity in that zone, one value per session.",
            graph=G_BAR, analysis=FDR_ZONE_NOTE)),
        "plot3_temporal_median": ("sessions_x_bins_x_zones", _xy_by_zone, dict(
            zone_col="node_type", x_col="time_bins", value_col="median_probability", x_title="Cumulative frames",
            what="Median probability of the Slow / Surveillance state (HMM state 1), one value per session per bin.",
            graph=G_POINTS, analysis=["none in the pipeline (a descriptive overlay) -- see Plot 2 for the "
                                       "zone-level significance testing."])),
        "plot3b_temporal_median_velocity": ("sessions_x_bins_x_zones", _xy_by_zone, dict(
            zone_col="node_type", x_col="time_bins", value_col="median_velocity", x_title="Cumulative frames",
            what="Median velocity across sessions, one value per session per bin.",
            graph=G_POINTS, analysis=["none in the pipeline (a descriptive overlay) -- see Plot 2b for the "
                                       "zone-level significance testing."])),
        "plot4_surveillance_bout": ("sessions_x_bout_outcome", _grouped_rows, dict(
            zone_col="successful_bout", value_col="probability_1", row_kind="bout outcome",
            what="Median surveillance probability (state 1) at the decision zone for that bout, one value per session.",
            graph=G_BAR, analysis=["Pipeline: within-genotype mixed model (Successful vs. Unsuccessful) with FDR, "
                                    "and across-genotype t-tests within each outcome -- see the Statistics table.",
                                    "In Prism: row 'Successful' vs. row 'Unsuccessful' as a paired/mixed comparison, "
                                    "or Multiple t tests across genotypes within one row."])),
        "plot5_bout_depth": ("sessions_x_lanes", _grouped_rows, dict(
            zone_col="deepest_lane", value_col="pct_bouts", row_kind="deepest lane",
            what="% of that session's bouts whose deepest point is in each lane (0 = entry lane, 7 = reward, "
                 "9 = terminal), one value per session per lane.",
            graph=G_BAR, analysis=["Pipeline: descriptive only (mean (SE) tables).",
                                   "In Prism: Multiple t tests (one per lane) with FDR correction, or a two-way "
                                   "ANOVA (lane x genotype)."])),
    },
}


def _resolve_zone_col(builder, kwargs, sheet):
    kwargs = dict(kwargs)
    if kwargs.get("zone_col") is None and "zone_col" in kwargs:
        inferred = _infer_zone_col(sheet, kwargs["value_col"])
        if inferred is None:
            return None, None
        kwargs["zone_col"] = inferred
    return builder, kwargs


def build_prism_store(project_path, config: dict, pipeline_sheets: dict) -> dict:
    """{f"{pipeline}/{step}": {"comparisons": [...], "zones": [...] | None,
    "tables": {(comparison, zone): table_dict}}} for every step with a
    STEP_SPECS entry and a matching sheet actually present in that
    pipeline's datasheet.

    pipeline_sheets : {pipeline_key: {sheet_name: DataFrame}} -- already
    read via source_data.read_sheets() for each pipeline's own workbook
    (each must include a "mice" sheet for Design).
    """
    genotype_sets = config.get("genotype_sets") or {}
    comparisons = [c for c in (config.get("comparisons") or []) if c in genotype_sets]
    store: dict = {}
    for pkey, specs in STEP_SPECS.items():
        sheets = pipeline_sheets.get(pkey)
        if not sheets or "mice" not in sheets:
            continue
        mice = sheets["mice"]
        designs = {cmp: Design(mice, genotype_sets[cmp]) for cmp in comparisons}
        for step, spec in specs.items():
            if spec is None:
                continue
            sheet_name, builder, kwargs = spec
            kwargs = dict(kwargs)
            where = kwargs.pop("where", None)
            sheet = sheets.get(sheet_name)
            if sheet is None or not len(sheet):
                continue
            # The step's columns may be missing if that step didn't run this
            # time (another step sharing the sheet did) -- skip, don't crash.
            needed = [kwargs[k] for k in ("value_col", "x_col") if kwargs.get(k)]
            if kwargs.get("zone_col"):
                needed.append(kwargs["zone_col"])
            if any(c not in sheet.columns for c in needed):
                continue
            # Only the primary (non-percentage) figures get Prism tables;
            # the "_pct" variants share the same sheets, at a different
            # x scale, so their rows are left out here.
            if "figure" in sheet.columns:
                sheet = sheet[~sheet["figure"].astype(str).str.endswith("_pct")
                              & ~sheet["figure"].astype(str).str.contains("_pct_")]
            # A sheet shared by several different metrics (several plotting
            # functions flushing under the same data-shape name) stacks
            # their rows together -- isolate this step's own rows before
            # grouping/pivoting, so a different metric's rows (NaN in this
            # step's value_col) don't leak in as spurious sessions/bins/zones.
            sheet = sheet.dropna(subset=[kwargs["value_col"]])
            # An extra {column: value} filter for a sheet shared by rows at
            # different scales, not just different metrics (e.g. Plot 6 vs.
            # 6b both flush "ind_no" into the same sheet, one as a raw bout
            # count and one as a % of that session's own bouts).
            if where:
                for col, value in where.items():
                    if col in sheet.columns:
                        sheet = sheet[sheet[col] == value]
            if not len(sheet):
                continue
            # One set of tables per zone set the step's figures were made
            # with (read off each row's "figure" name, e.g. "..._9_corners"),
            # so the report's zone-set selector picks the matching table.
            if "figure" in sheet.columns:
                zone_sets = sheet["figure"].astype(str).map(lambda f: zone_set_of(f, config) or "_")
            else:
                zone_sets = pd.Series("_", index=sheet.index)
            sheet = sheet.assign(_zone_set=zone_sets)
            tables: dict = {}
            regions: dict = {}
            for zone_set, sub in sheet.groupby("_zone_set", sort=False):
                builder_zs, resolved_kwargs = _resolve_zone_col(builder, kwargs, sub)
                if builder_zs is None:
                    continue
                for cmp in comparisons:
                    design = designs[cmp]
                    if not design.groups or design.n_max == 0:
                        continue
                    for region, table in builder_zs(sub, design, **resolved_kwargs):
                        region_key = region if region is not None else "_"
                        regions.setdefault(zone_set, [])
                        if region_key not in regions[zone_set]:
                            regions[zone_set].append(region_key)
                        tables[(cmp, zone_set, region_key)] = table
            if not tables:
                continue
            store[f"{pkey}/{step}"] = {
                "comparisons": [c for c in comparisons if any(k[0] == c for k in tables)],
                # Per zone set, the regions (e.g. the zones of an "all regions"
                # overlay) its tables are split by -- None when not split.
                "regions": {zs: (r if r != ["_"] else None) for zs, r in regions.items()},
                "tables": tables,
            }
    return store
