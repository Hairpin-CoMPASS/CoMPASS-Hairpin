"""The values behind every figure, collected while a post-analysis notebook
runs and written at the end as one datasheet workbook for the whole
pipeline (each helper module's write_*_datasheet()).

A plotting function hands over what it drew with record(df); the code that
saves the figure calls flush(name), which keeps the tables under that
figure's name. Each table gets the mouse's metadata (mouse_id, group,
design factors, sex) in front, joined on its "session" column, plus any
lookup columns the configured annotators add (e.g. motif communities).
Tables live in this module, so they survive reloading the helper modules;
rerunning a step replaces its tables.
"""

import os
import pickle

import pandas as pd

# Columns plotting code carries internally; the metadata join supplies the real ones.
_INTERNAL = {"group_label", "group_label_n", "genotype", "treatment", "category"}
# Sheets past this many rows go to their own CSV beside the workbook.
MAX_SHEET_ROWS = 500_000

_PENDING = []
_TABLES = {}
_CONTEXT = {"meta": None, "columns": [], "annotators": []}


def configure(meta, factor_columns=(), annotators=None):
    """Metadata joined onto every table: `meta` needs session and
    group_label; mouse_id, `factor_columns`, sex and timepoint are added
    when present. `annotators`: functions df -> df adding lookup columns."""
    meta = meta.copy()
    meta["session"] = meta["session"].astype(str)
    columns = ["session"] + [c for c in ("mouse_id",) if c in meta.columns] + ["group_label"]
    columns += [c for c in (*factor_columns, "sex", "timepoint") if c in meta.columns and c not in columns]
    meta = meta[columns].rename(columns={"group_label": "group"})
    previous = _CONTEXT["meta"]
    if previous is not None and list(previous.columns) == list(meta.columns):
        meta = pd.concat([previous, meta]).drop_duplicates("session", keep="last")
    _CONTEXT["meta"] = meta.reset_index(drop=True)
    _CONTEXT["columns"] = list(meta.columns)
    if annotators is not None:
        _CONTEXT["annotators"] = list(annotators)


def metadata():
    """One row per mouse: the configured metadata columns."""
    return _CONTEXT["meta"].copy()


def record(df, part=""):
    """Queue `df` for the next flush(). `part` names an extra table kept
    beside the figure's main one (<name>_<part>); same-part tables are stacked."""
    _PENDING.append((part, df.copy()))


def discard():
    _PENDING.clear()


def annotate(df):
    """`df` with the metadata columns in front and the annotators' columns."""
    meta = _CONTEXT["meta"]
    df = df.drop(columns=[c for c in df.columns if c in _INTERNAL])
    if meta is not None and "session" in df.columns:
        df = df.copy()
        df["session"] = df["session"].astype(str)
        payload = [c for c in df.columns if c not in _CONTEXT["columns"] or c == "session"]
        df = meta.merge(df[payload], on="session", how="right")
    for annotator in _CONTEXT["annotators"]:
        df = annotator(df)
    return df


def keep(name, df):
    """Keep `df` (annotated) under `name` now."""
    _TABLES[name] = annotate(df)
    return name


def flush(name, figure=None):
    """Keep the queued tables under `name` (<name>_<part> for parts).

    Without `figure`, the tables replace whatever `name` held. With `figure`
    (the saved figure's file stem), they are added to it instead, tagged
    with a "figure" column: several figures can share one sheet, and
    rerunning a figure replaces only that figure's own rows."""
    if not _PENDING:
        return []
    tables = {}
    for part, df in _PENDING:
        key = f"{name}_{part}" if part else name
        tables[key] = pd.concat([tables[key], df], ignore_index=True) if key in tables else df
    discard()
    if figure is None:
        return [keep(key, df) for key, df in tables.items()]
    for key, df in tables.items():
        new = annotate(df).assign(figure=figure)
        old = _TABLES.get(key)
        if old is not None and "figure" in old.columns:
            old = old[old["figure"] != figure]
            new = pd.concat([old, new], ignore_index=True)
        _TABLES[key] = new
    return list(tables)


def get(name):
    """A kept table, or None if that step hasn't run in this session."""
    return _TABLES.get(name)


def find(prefix):
    """{name: table} of every kept table whose name starts with `prefix`."""
    return {k: v for k, v in _TABLES.items() if k.startswith(prefix)}


def write_workbook(path, sheets, readme):
    """Write `sheets` ({sheet name: DataFrame or None}) to the Excel workbook
    `path`, after a README sheet built from `readme` ({sheet: (description,
    figures)}). Sheets too long for Excel go to <workbook stem>_<sheet>.csv
    beside it. Returns the written paths."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = os.path.splitext(path)[0]
    present = {k: v for k, v in sheets.items() if v is not None and len(v)}
    big = {k for k, v in present.items() if len(v) > MAX_SHEET_ROWS}
    rows = []
    for sheet, df in present.items():
        description, figures = readme.get(sheet, ("", ""))
        where = f"{os.path.basename(stem)}_{sheet}.csv (too long for Excel)" if sheet in big else "this workbook"
        rows.append({"sheet": sheet, "one row per": description, "figures": figures, "rows": len(df),
                     "columns": ", ".join(map(str, df.columns)), "location": where})
    missing = [s for s, v in sheets.items() if v is None or not len(v)]
    if missing:
        rows.append({"sheet": "(not in this run)", "one row per": ", ".join(missing),
                     "figures": "run those steps, then this cell again", "rows": 0, "columns": "", "location": ""})
    written = [path]
    try:
        open(path, "a").close()  # fails early if Excel (or another program) has the workbook open
    except PermissionError:
        raise PermissionError(f"Can't write {path}: it is open in another program (usually Excel). Close it there, "
                              f"then run this cell again.") from None
    in_workbook = {"README": pd.DataFrame(rows)}
    in_workbook.update({sheet[:31]: df for sheet, df in present.items() if sheet not in big})
    with pd.ExcelWriter(path) as writer:
        for sheet, df in in_workbook.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    for sheet in sorted(big):
        csv = f"{stem}_{sheet}.csv"
        present[sheet].to_csv(csv, index=False, float_format="%.6g")
        written.append(csv)
    # The same sheets in a file that loads in a second, for the report (read_sheets()); it is ignored
    # once the workbook is newer (e.g. edited in Excel).
    with open(f"{stem}{FAST_COPY}", "wb") as f:
        pickle.dump(in_workbook, f)
    print("saved datasheet", "\n  ".join(written))
    return written


FAST_COPY = ".sheets.pkl"


def read_sheets(path, dtype=None):
    """Every sheet of datasheet `path` ({name: DataFrame}), from its fast copy
    when that is at least as new as the workbook, else from the workbook.
    `dtype`: {column: type} applied as read_excel would."""
    fast = os.path.splitext(path)[0] + FAST_COPY
    if os.path.exists(fast) and os.path.getmtime(fast) >= os.path.getmtime(path):
        with open(fast, "rb") as f:
            sheets = pickle.load(f)
        for df in sheets.values():
            for col, kind in (dtype or {}).items():
                if col in df.columns:
                    df[col] = df[col].map(lambda v: v if pd.isna(v) else kind(v)).astype(object)
        return sheets
    return pd.read_excel(path, sheet_name=None, dtype=dtype)
