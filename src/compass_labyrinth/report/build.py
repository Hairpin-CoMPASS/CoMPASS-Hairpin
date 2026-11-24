"""Builds the self-contained hairpin-maze figure report.

Reads figures from `figures/<pipeline>/<step_folder>/*.{pdf,png}` and, when
present, a matching stats folder at `results/<pipeline>/<step_folder>/*.csv`,
and writes one offline-viewable `analysis_report.html` at the project root.
Every figure preview is embedded as a base64 WebP data URI so the page opens
without network access from a synced Dropbox/OneDrive folder; the "Open
file" links next to each figure are plain relative paths to the original
PDF/PNG, which only work where the project folder itself is present.

Call `build_report(project_path)` from a notebook's last cell or a small
script after the figure-producing cells have run.
"""
from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pymupdf
import yaml
from PIL import Image

from compass_labyrinth.report.figure_descriptions import PIPELINES, REPORT_TITLE
from compass_labyrinth.report.prism_tables import build_prism_store, detect_key, zone_set_of
import compass_labyrinth.source_data as source_data
import compass_labyrinth.figure_labels as figure_labels


def _comparison_label(set_name: str, genotype_sets: dict) -> str:
    labels = list(genotype_sets[set_name].keys())
    return " vs. ".join(labels) if len(labels) <= 2 else f"{len(labels)} genotypes ({', '.join(labels)})"


def _load_comparisons(project_path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    """Read this project's own config.yaml -- genotype_sets and comparisons --
    so nothing genotype-specific is hard-coded here; a project with neither
    key set (or none at all) just gets no comparison selector anywhere. This
    is the SAME config.yaml the notebooks read ZONES/ZONE_SETS/GENOTYPE_SETS/
    COMPARISONS from, so there is exactly one place to edit per project.
    """
    path = project_path / "config.yaml"
    if not path.exists():
        return (), {}
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    comparisons = config.get("comparisons") or []
    genotype_sets = config.get("genotype_sets") or {}
    comparisons = [c for c in comparisons if c in genotype_sets]
    # Longest (multi-word) keys first, so e.g. a hypothetical "wt_nlf_pair"
    # key wouldn't be shadowed by a shorter key matching part of it first.
    keys = tuple(sorted(comparisons, key=len, reverse=True))
    labels = {c: _comparison_label(c, genotype_sets) for c in comparisons}
    return keys, labels

def _prism_for_step(prism_store: dict, pkey: str, folder: str) -> dict | None:
    """This step's Prism tables for the report JSON: {"regions": {zone set:
    [region, ...] | None}, "tables": {"<comparison>|<zone set>|<region>":
    table}} -- JSON-safe (string keys), or None if the step has no Prism
    support. The comparison and zone set come from the step's own selectors
    (the same ones choosing which figures show); the region, when a table is
    split by one, is picked inside the Prism card."""
    entry = prism_store.get(f"{pkey}/{folder}")
    if entry is None:
        return None
    return {
        "regions": entry["regions"],
        "tables": {f"{cmp}|{zs}|{region}": table for (cmp, zs, region), table in entry["tables"].items()},
    }


def _find_datasheet(project_path: Path, pkey: str) -> str | None:
    """This pipeline's datasheet workbook (written by the notebook's "Write
    the Data Sheet" cell), as a path relative to the report -- for a direct
    download link and a "show its folder" button, or None if it hasn't been
    written yet."""
    results_dir = project_path / "results" / pkey
    matches = sorted(results_dir.glob("*_datasheet.xlsx")) if results_dir.exists() else []
    return f"results/{pkey}/{matches[0].name}" if matches else None


PREVIEW_LONG_SIDE = 2000  # px -- fixed, no per-folder tuning (this project is small)
FIT_MAX_TALLNESS = 1.5  # height / width above which a figure isn't fit to the screen
MIN_SHORT_SIDE = 900  # px floor for a figure's narrower dimension -- an unusually
# elongated PDF (many stacked subplot rows, say) would otherwise have its short
# side scaled down along with the long one, leaving it blurry/illegible even
# though PREVIEW_LONG_SIDE itself is generous. A normal (roughly landscape or
# portrait) figure's short side already exceeds this once scaled by the long
# side alone, so this only kicks in for the unusual case.
WEBP_QUALITY = 95
FIGURE_EXTS = {".pdf", ".png"}
MAZE_SVG_PATH = Path(__file__).resolve().parents[3] / "media" / "new_maze_layout_v2.svg"


def _preview_scale(width: float, height: float) -> float:
    scale_long = PREVIEW_LONG_SIDE / max(width, height)
    scale_short = MIN_SHORT_SIDE / min(width, height)
    return max(scale_long, scale_short)


def _render_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        doc = pymupdf.open(str(path))
        page = doc[0]
        scale = _preview_scale(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return img
    img = Image.open(path).convert("RGB")
    scale = _preview_scale(img.width, img.height)
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img


def _to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=WEBP_QUALITY)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _title_from_stem(stem: str) -> str:
    return stem.replace("_", " ").strip()


_SIG_FDR_RE = re.compile(r"(?i)^q([_-]?value)?$")
_SIG_P_RE = re.compile(r"(?i)(^|[_-])p([_-]?value)?$")


def _find_sig_column(columns: list[str]) -> str | None:
    for c in columns:
        if _SIG_FDR_RE.match(c) or "fdr" in c.lower():
            return c
    for c in columns:
        if _SIG_P_RE.search(c):
            return c
    return None


def _collect_stats(stats_dir: Path, project_path: Path, comparison_keys: tuple[str, ...], config: dict) -> list[dict]:
    tables = []
    if not stats_dir.is_dir():
        return tables
    for csv_path in sorted(stats_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        sig_col = _find_sig_column(list(df.columns))
        rows = json.loads(df.to_json(orient="records"))
        tables.append(
            {
                "name": _title_from_stem(csv_path.stem),
                "columns": list(df.columns),
                "rows": rows,
                "sigCol": sig_col,
                "file": csv_path.relative_to(project_path).as_posix(),
                # Shown only under the matching Comparison / Zone set selection
                # (a table naming neither shows under every selection).
                "comparison": detect_key(csv_path.stem, comparison_keys),
                "zoneSet": zone_set_of(csv_path.stem, config),
            }
        )
    return tables


def _collect_items(fig_dir: Path, uid_prefix: str, images: dict, comparison_keys: tuple[str, ...],
                   config: dict) -> list[dict]:
    items = []
    if not fig_dir.is_dir():
        return items
    files = sorted(p for p in fig_dir.iterdir() if p.suffix.lower() in FIGURE_EXTS)
    for i, f in enumerate(files):
        # An empty/half-written file (e.g. still syncing) is skipped, not fatal
        try:
            img = _render_image(f)
        except Exception as e:
            print(f"Skipped unreadable figure {f.name} ({type(e).__name__}: {e}) -- rebuild the report once it's written")
            continue
        image_id = f"{uid_prefix}-{i}"
        images[image_id] = _to_data_uri(img)
        items.append(
            {
                "title": _title_from_stem(f.stem),
                "file": f.name,
                "image": image_id,
                "size": list(img.size),
                # Fit to the screen height unless the figure is a long stack
                # (more than FIT_MAX_TALLNESS x taller than wide, e.g. task
                # performance Plot 1's many time-bin rows), which keeps full
                # width and scrolls
                "fit": img.size[1] <= FIT_MAX_TALLNESS * img.size[0],
                # Read off the file name (e.g. "..._wt_nlf_9_corners.pdf"):
                # which Comparison / Zone set selection this figure belongs to.
                "comparison": detect_key(f.stem, comparison_keys),
                "zoneSet": zone_set_of(f.stem, config),
                "labels": figure_labels.read(str(f)) if f.suffix.lower() == ".pdf" else None,
            }
        )
    return items


def _load_prism_sheets(project_path: Path) -> dict:
    """{pipeline_key: {sheet_name: DataFrame}} for every pipeline whose
    datasheet workbook exists on disk -- the per-session values prism_tables
    reshapes into Prism-ready tables. A pipeline with no datasheet yet (the
    notebook's "Write the Data Sheet" cell hasn't run) is just missing from
    the dict; build_prism_store() skips it."""
    sheets = {}
    for pkey in PIPELINES:
        rel = _find_datasheet(project_path, pkey)
        if rel is None:
            continue
        try:
            sheets[pkey] = source_data.read_sheets(str(project_path / rel))
        except Exception as e:
            print(f"Prism tables skipped for {pkey}: could not read its datasheet ({type(e).__name__}: {e})")
    return sheets


def _build_data(project_path: Path, images: dict) -> dict:
    comparison_keys, comparison_labels = _load_comparisons(project_path)
    config_path = project_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    prism_store = build_prism_store(project_path, config, _load_prism_sheets(project_path))
    pipelines = []
    sig_rows = []
    for pkey, pcfg in PIPELINES.items():
        figures_root = project_path / "figures" / pkey
        results_root = project_path / "results" / pkey
        steps = []
        n_figs = 0
        for step_cfg in pcfg["steps"]:
            folder = step_cfg["folder"]
            uid = f"{pkey}-{folder}"
            items = _collect_items(figures_root / folder, uid, images, comparison_keys, config)
            if step_cfg.get("fit") is False:  # step opts out of fit-to-window: full width, scrolls
                for item in items:
                    item["fit"] = False
            if step_cfg.get("width"):  # step-level figure width, % of the card (e.g. 50)
                for item in items:
                    item["width"] = step_cfg["width"]
            # Maze heatmaps: fit the card's width (full width, scroll down)
            # rather than being shrunk to the window height
            if step_cfg.get("grid_sizing"):
                for item in items:
                    item["fit"] = False
                    item["width"] = None
            stats = _collect_stats(results_root / folder, project_path, comparison_keys, config)
            # Optional extra selectors for this step only (e.g. Plot 7's
            # direction, the heatmaps' time bins): each figure/table is tagged
            # with the option key in its file name; only options some figure
            # has get a pill. A selector with a "none_label" files untagged
            # figures under "none" (listed first, the default) instead of
            # showing them under every option.
            selectors = []
            cfg_selectors = step_cfg.get("selectors") or ([step_cfg["selector"]] if step_cfg.get("selector") else [])
            for sel in cfg_selectors:
                skey = sel["label"]  # same label on different steps = one shared choice
                keys = [k for k, _ in sel["options"]]
                for entry in items + stats:
                    key = detect_key(Path(entry["file"]).stem, keys)
                    if key is None and sel.get("none_label"):
                        key = "none"
                    entry.setdefault("options", {})[skey] = key
                labels = ([("none", sel["none_label"])] if sel.get("none_label") else []) + list(sel["options"])
                options = [{"key": k, "label": lab} for k, lab in labels
                           if any(it["options"][skey] == k for it in items)]
                # Only "All frames" present (nothing binned yet) -> no row
                if sel.get("none_label") and [o["key"] for o in options] == ["none"]:
                    continue
                selectors.append({"key": skey, "label": sel["label"], "options": options})
            n_figs += len(items)
            for table in stats:
                if not table["sigCol"]:
                    continue
                for row in table["rows"]:
                    val = row.get(table["sigCol"])
                    if isinstance(val, (int, float)) and val < 0.05:
                        sig_rows.append(
                            {
                                "pipeline": pkey,
                                "pipelineName": pcfg["name"],
                                "step": folder,
                                "stepTitle": step_cfg["title"],
                                "table": table["name"],
                                "sigCol": table["sigCol"],
                                "value": val,
                                "row": row,
                            }
                        )
            # comparison_labels is in config.yaml's declared order (comparison_keys
            # is longest-first, for filename matching) -- pills follow the config.
            present = [k for k in comparison_labels if any(it["comparison"] == k for it in items)]
            prism_entry = prism_store.get(f"{pkey}/{folder}")
            # Only the comparisons this step actually generated figures for
            # get a pill (even just one -- shown selected, for reference);
            # the Prism table follows the same selection.
            comparisons = [{"key": k, "label": comparison_labels[k]} for k in present]
            # Untagged figures: the Prism table uses config.yaml's first comparison.
            if present:
                default_comparison = present[0]
            elif prism_entry and prism_entry["comparisons"]:
                default_comparison = prism_entry["comparisons"][0]
            else:
                default_comparison = None
            # The Comparison row is always shown: untagged figures still list
            # the one comparison their Prism table uses, selected.
            if not comparisons and default_comparison:
                comparisons = [{"key": default_comparison, "label": comparison_labels[default_comparison]}]
            # Zone sets this step's figures were made with, in config.yaml
            # order -- one pill each, even a single one (selected).
            zone_set_order = list(config.get("zone_sets") or {}) + list(config.get("zones") or {})
            zone_sets = [z for z in zone_set_order if any(it["zoneSet"] == z for it in items)]
            # A step whose plain (per-cell) figures sit alongside zone-set
            # versions can offer "none" -- first, so the plain figures stay
            # the default -- instead of showing them under every zone set
            if step_cfg.get("zone_set_none") and zone_sets:
                for entry in items + stats:
                    if entry.get("zoneSet") is None:
                        entry["zoneSet"] = "none"
                zone_sets = ["none"] + zone_sets
            steps.append(
                {
                    "id": folder,
                    "title": step_cfg["title"],
                    "about": {"summary": step_cfg["summary"], "items": step_cfg.get("items", [])},
                    "items": items,
                    "stats": stats,
                    "comparisons": comparisons,
                    "defaultComparison": default_comparison,
                    "zoneSets": zone_sets,
                    "selectors": selectors,
                    "prism": _prism_for_step(prism_store, pkey, folder),
                }
            )
        pipelines.append({
            "key": pkey, "name": pcfg["name"], "steps": steps, "figureCount": n_figs,
            "datasheet": _find_datasheet(project_path, pkey),
        })
    sig_rows.sort(key=lambda r: r["value"])
    maze_svg = MAZE_SVG_PATH.read_text(encoding="utf-8") if MAZE_SVG_PATH.exists() else None
    # For the zone-set reference page: every zone set / zone in config.yaml,
    # each zone's grid cells, and which steps' figures use each set
    used_by: dict[str, list[str]] = {}
    for p in pipelines:
        for s in p["steps"]:
            for z in s["zoneSets"]:
                if z == "none":  # the plain per-cell view, not a zone set
                    continue
                used_by.setdefault(z, []).append(f"{p['name']} · {s['title']}")
    zone_info = {
        "sets": {k: list(v) for k, v in (config.get("zone_sets") or {}).items()},
        "zones": {k: list(v) for k, v in (config.get("zones") or {}).items()},
        "usedBy": used_by,
    }
    return {
        "zoneInfo": zone_info,
        "projectName": project_path.parent.name if project_path.name == "compass_analysis" else project_path.name,
        "title": REPORT_TITLE,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mazeSvg": maze_svg,
        "pipelines": pipelines,
        "significant": sig_rows,
    }


def _render_html(data: dict, images: dict) -> str:
    # Defensive: a literal "</script" inside any embedded string (e.g. the
    # maze SVG, or a future figure title) would otherwise close the tag early.
    data_json = json.dumps(data).replace("</script", "<\\/script")
    image_scripts = "\n".join(
        f'<script type="text/plain" id="img-{html.escape(k)}">{v}</script>' for k, v in images.items()
    )
    return _TEMPLATE.format(
        data_json=data_json,
        image_scripts=image_scripts,
    )


def build_report(project_path: str | Path) -> Path:
    project_path = Path(project_path)
    images: dict[str, str] = {}
    data = _build_data(project_path, images)
    out_html = _render_html(data, images)
    out_path = project_path / "analysis_report.html"
    out_path.write_text(out_html, encoding="utf-8")
    n_figs = sum(p["figureCount"] for p in data["pipelines"])
    print(f"Wrote {out_path} ({n_figs} figures, {len(data['significant'])} significant rows)")
    return out_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hairpin maze analysis report</title>
<style>
:root {{
  --bg: #ffffff; --panel: #f6f7f9; --border: #dde1e6; --text: #1b1f24; --muted: #5b6370;
  --accent: #2563eb; --accent-contrast: #ffffff; --sig: #b91c1c; --sig-bg: #fef2f2;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; height: 100%; }}
body {{ font: 14px/1.45 -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: var(--text); background: var(--bg); display: flex; flex-direction: column; }}
header {{ padding: 10px 16px 0; border-bottom: 1px solid var(--border); background: var(--panel); }}
.brand {{ padding-bottom: 8px; }}
.brand h1 {{ font-size: 17px; margin: 0; }}
.brand small {{ color: var(--muted); }}
.toprow {{ display: flex; gap: 12px; align-items: center; padding-bottom: 10px; flex-wrap: wrap; }}
#search {{ width: 340px; max-width: 40vw; padding: 7px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; }}
.tabs {{ display: flex; gap: 8px; flex: 1; }}
.tab-btn {{ flex: 1; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border); background: #fff; cursor: pointer; font-size: 14px; }}
.tab-btn.active {{ background: var(--accent); color: var(--accent-contrast); border-color: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.2); }}
.tab-mini {{ flex: none; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border); background: #fff; cursor: pointer; font-size: 14px; white-space: nowrap; }}
.tab-mini.active {{ background: var(--accent); color: var(--accent-contrast); border-color: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.2); }}
.bmbtn {{ flex: none; width: 260px; border: 2px solid #c58a00; background: #fff; color: #9a6a00; border-radius: 8px; font: inherit; font-weight: 700; font-size: 14px; cursor: pointer; padding: 6px 12px; display: flex; flex-direction: column; justify-content: center; line-height: 1.25; }}
.bmbtn small {{ font-weight: 500; font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bmbtn:hover:not(:disabled) {{ background: #fff4d6; }}
.bmbtn:disabled {{ opacity: .55; cursor: default; }}
#layout {{ flex: 1; display: flex; min-height: 0; }}
#leftbar {{ width: 330px; flex-shrink: 0; border-right: 1px solid var(--border); padding: 10px; overflow-y: auto; }}
#rightbar {{ width: 280px; flex-shrink: 0; border-left: 1px solid var(--border); padding: 10px; overflow-y: auto; }}
#main {{ flex: 1; padding: 16px 20px; min-width: 0; overflow-y: auto; }}
.nav-fixed-item, .nav-step {{ display: block; width: 100%; text-align: left; background: none; border: none; padding: 7px 8px; border-radius: 6px; cursor: pointer; font-size: 13.5px; color: var(--text); white-space: normal; word-break: break-word; }}
.nav-fixed-item:hover, .nav-step:hover {{ background: #eef1f5; }}
.nav-step.active {{ background: var(--accent); color: var(--accent-contrast); }}
.nav-group-title {{ font-size: 11px; font-weight: 700; letter-spacing: .04em; color: var(--muted); margin: 14px 4px 4px; }}
.pathbox {{ position: relative; margin: 2px 4px 6px; padding: 6px 8px; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; font-size: 12px; color: var(--muted); }}
.pathbox .pathclose {{ position: absolute; top: 2px; right: 4px; border: 0; background: none; color: var(--muted); cursor: pointer; font-size: 13px; padding: 2px; }}
.pathbox .pathclose:hover {{ color: var(--text); }}
.pathbox code {{ display: block; color: var(--text); word-break: break-all; user-select: all; margin-top: 2px; }}
.search-group {{ margin-bottom: 10px; }}
.search-group h4 {{ margin: 6px 4px; font-size: 12px; color: var(--muted); }}
h1.step-heading {{ font-size: 19px; margin: 0 0 2px; }}
.step-count {{ color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
.controls {{ background: var(--bg); margin: -16px -20px 10px; padding: 16px 20px 10px; border-bottom: 1px solid var(--border); }}
.cmp-row {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 6px 0; }}
.cmp-label {{ color: var(--muted); margin-right: 4px; font-size: 13px; min-width: 84px; }}
a.cmp-link {{ text-decoration: underline dotted; cursor: pointer; }}
a.cmp-link:hover {{ color: var(--accent, #2563eb); }}
.back-btn {{ margin-bottom: 8px; padding: 4px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); cursor: pointer; font-size: 13px; }}
.zs-card {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; }}
.zs-card.zs-current {{ border-color: var(--accent, #2563eb); box-shadow: 0 0 0 2px rgba(37, 99, 235, .15); }}
.zs-head {{ margin-bottom: 6px; }}
.zs-name {{ font: 600 14px ui-monospace, Consolas, monospace; padding: 2px 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); cursor: pointer; color: inherit; }}
.zs-name:hover {{ border-color: var(--accent, #2563eb); }}
.zs-name.copied {{ border-color: #16a34a; }}
.zs-copy-hint {{ font: 11px system-ui, sans-serif; color: var(--muted); margin-left: 8px; }}
.zs-name.copied .zs-copy-hint {{ color: #16a34a; }}
.zs-meta {{ color: var(--muted); font-size: 12px; margin-left: 8px; }}
.zs-tag {{ font-size: 11px; padding: 1px 6px; border-radius: 8px; background: var(--accent, #2563eb); color: #fff; }}
.zs-table {{ border-collapse: collapse; font-size: 13px; }}
.zs-table th, .zs-table td {{ text-align: left; padding: 3px 14px 3px 0; border-bottom: 1px solid var(--border); vertical-align: top; }}
.zs-table th {{ color: var(--muted); font-weight: 600; }}
.zs-used {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
.cmp-none {{ color: var(--muted); font-size: 13px; font-style: italic; }}
.pill {{ border: 1px solid var(--border); background: #fff; color: var(--text); border-radius: 999px; padding: 5px 12px; font: inherit; font-size: 13px; cursor: pointer; }}
.pill.active {{ background: var(--accent); border-color: var(--accent); color: var(--accent-contrast); }}
.pill:disabled {{ opacity: 0.35; cursor: not-allowed; }}
.card {{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 18px; overflow: hidden; }}
.card-title {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--border); }}
.card-title b {{ flex: 1; font-weight: 600; font-size: 13.5px; }}
.card-title a, .card-title button {{ font-size: 12.5px; color: var(--accent); background: none; border: none; cursor: pointer; text-decoration: none; }}
.card img {{ display: block; max-width: 100%; width: auto; height: auto; margin: 0 auto; cursor: zoom-in; }}
.card img.fit {{ max-height: var(--fit-h, calc(100vh - 320px)); }}
.card img[data-w] {{ max-width: var(--w); }}
.card.actual-size img {{ max-width: none; max-height: none; }}
details.dd {{ border-top: 1px solid var(--border); }}
details.dd summary {{ cursor: pointer; padding: 7px 12px; font-size: 13px; color: var(--muted); }}
details.dd .dd-body {{ padding: 2px 12px 12px; font-size: 13px; }}
.dd-body .lead {{ color: var(--text); }}
.about-body dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 6px 0 0; }}
.about-body dt {{ font-weight: 600; color: var(--muted); white-space: nowrap; }}
.about-body dd {{ margin: 0; }}
.about-body ul {{ margin: 2px 0; padding-left: 18px; }}
table.stats {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin-bottom: 8px; }}
table.stats th, table.stats td {{ border: 1px solid var(--border); padding: 4px 6px; text-align: left; }}
table.stats tr.sig {{ background: var(--sig-bg); color: var(--sig); }}
.toggle-sig {{ font-size: 12px; margin-bottom: 6px; }}
.prism-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }}
.prism-status {{ color: var(--muted); font-size: 12px; }}
.prism-about {{ margin-bottom: 6px; }}
.prism-step {{ margin-top: 4px; }}
.prism-sub {{ margin: 2px 0 2px 0; padding-left: 34px; }}
.prism-sub li {{ margin: 1px 0; }}
.prism-foot {{ margin-top: 8px; font-size: 12px; font-style: italic; color: var(--muted); }}
.prism-titles {{ margin-top: 8px; }}
.prism-line {{ margin-left: 12px; white-space: pre-wrap; font-size: 13px; }}
.prism-key {{ color: var(--muted); }}
.prism-panel {{ font-style: italic; margin-top: 2px; }}
.bookmark-item {{ border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin-bottom: 8px; font-size: 12.5px; cursor: pointer; position: relative; }}
.bookmark-item:hover {{ background: #eef1f5; }}
.bookmark-item .bm-del {{ position: absolute; top: 6px; right: 8px; color: #c0392b; background: none; border: none; cursor: pointer; }}
.bookmark-item.stale {{ opacity: .5; }}
.warn-box {{ background: #fff8e6; border: 1px solid #f0d58c; border-radius: 6px; padding: 10px; font-size: 12px; color: #7a5a00; margin-top: 10px; }}
.warn-box ul {{ margin: 6px 0 0; padding-left: 16px; }}
.warn-box li {{ margin: 3px 0; }}
.sig-row-link {{ display: block; padding: 8px; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 8px; cursor: pointer; font-size: 12.5px; }}
.sig-row-link:hover {{ background: #eef1f5; }}
.highlight-flash {{ outline: 3px solid #f5a623; outline-offset: 2px; }}
.maze-wrap {{ text-align: center; }}
.maze-wrap svg {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<header>
  <div class="brand"><h1 id="title"></h1><small id="generated"></small></div>
  <div class="toprow">
    <input id="search" type="text" placeholder="Find a step...">
    <div class="tabs" id="tabs"></div>
    <button id="bmbtn" class="bmbtn" disabled>☆ Bookmark this figure<small id="bmcur"></small></button>
  </div>
</header>
<div id="layout">
  <nav id="leftbar"></nav>
  <main id="main"></main>
  <aside id="rightbar">
    <h3 style="margin-top:0">Bookmarks</h3>
    <div id="bookmark-list"></div>
    <div class="warn-box">
      <b>Bookmarks are saved in this browser only.</b>
      <ul>
        <li>Clearing "cookies and site data", cleanup tools, or "clear data on close" settings delete them.</li>
        <li>Private / incognito windows never keep them.</li>
        <li>Another browser or computer has its own, separate bookmarks.</li>
      </ul>
    </div>
  </aside>
</div>
{image_scripts}
<script type="application/json" id="report-data">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
// mode defaults to 'maze' -- every fresh open (new tab/window, or reopening
// the file later) starts there. perPipeline remembers each pipeline's last
// step+comparison so switching tabs restores where you were; sessionStorage
// (not localStorage) means a browser refresh of the SAME tab keeps the
// current page exactly as-is, but a genuinely new tab/window starts fresh.
let state = {{ pipeline: DATA.pipelines[0].key, step: null, query: '', mode: 'maze', comparison: null,
  zoneSet: null, options: {{}}, prismRegion: null, prismOpen: false, perPipeline: {{}} }};
document.getElementById('title').textContent = DATA.title;
document.getElementById('generated').textContent = `Report generated at ${{DATA.generated}}`;

function imgSrc(id) {{
  const el = document.getElementById('img-' + id);
  return el ? el.textContent : '';
}}

function loadViewState() {{
  try {{
    const raw = sessionStorage.getItem('hairpin-report-view');
    if (raw) Object.assign(state, JSON.parse(raw));
  }} catch (e) {{}}
}}
function saveViewState() {{
  try {{
    sessionStorage.setItem('hairpin-report-view', JSON.stringify({{
      mode: state.mode, pipeline: state.pipeline, perPipeline: state.perPipeline,
    }}));
  }} catch (e) {{}}
}}

// --- Bookmarks: kept in this browser per project (survive closing the page and rebuilding the report) ---
const BOOKMARK_KEY = 'hairpin-report-bookmarks:' + (DATA.projectName || 'default');
let bookmarks = (() => {{ try {{ return JSON.parse(localStorage.getItem(BOOKMARK_KEY)) || []; }} catch (e) {{ return []; }} }})();
const saveBookmarks = () => {{ try {{ localStorage.setItem(BOOKMARK_KEY, JSON.stringify(bookmarks)); }} catch (e) {{}} }};
const allItems = () => DATA.pipelines.flatMap(p => p.steps.flatMap(s => s.items.map(i => ({{p, s, i}}))));
const isBookmarked = (pipelineKey, stepId, file) => bookmarks.some(b => b.pipeline === pipelineKey && b.step === stepId && b.file === file);

function currentCard() {{
  const main = document.getElementById('main');
  if (state.mode !== 'step' || !state.step) return null;
  const top = main.getBoundingClientRect().top + 40;
  return [...main.querySelectorAll('.card[data-file]')].find(c => c.getBoundingClientRect().bottom > top) || null;
}}

function updateBookmarkButton() {{
  const btn = document.getElementById('bmbtn'), cur = document.getElementById('bmcur');
  const card = currentCard();
  if (!card) {{ btn.disabled = true; btn.firstChild.textContent = '☆ Bookmark this figure'; cur.textContent = 'open a figure page'; return; }}
  const done = isBookmarked(state.pipeline, state.step, card.dataset.file);
  btn.disabled = done;
  btn.firstChild.textContent = done ? '★ Bookmarked' : '☆ Bookmark this figure';
  cur.textContent = card.dataset.title;
}}

function addBookmark() {{
  const card = currentCard();
  if (!card || isBookmarked(state.pipeline, state.step, card.dataset.file)) return;
  const p = DATA.pipelines.find(p => p.key === state.pipeline);
  const s = p.steps.find(s => s.id === state.step);
  bookmarks.unshift({{
    pipeline: state.pipeline, step: state.step, file: card.dataset.file, title: card.dataset.title,
    pipelineName: p.name, stepTitle: s.title, added: new Date().toISOString().slice(0, 10),
  }});
  saveBookmarks();
  renderBookmarks();
  updateBookmarkButton();
}}

function renderBookmarks() {{
  const el = document.getElementById('bookmark-list');
  if (!bookmarks.length) {{
    el.innerHTML = '<p style="color:#888;font-size:12.5px">No bookmarks yet. Scroll to a figure and click "Bookmark this figure" above.</p>';
    return;
  }}
  el.innerHTML = bookmarks.map((b, k) => {{
    const here = allItems().some(x => x.p.key === b.pipeline && x.s.id === b.step && x.i.file === b.file);
    return `<div class="bookmark-item ${{here ? '' : 'stale'}}" data-k="${{k}}" title="${{here ? 'Go to this figure' : 'Not in this version of the report'}}">
      <div>${{escapeHtml(b.title)}}</div>
      <div style="color:#888">${{escapeHtml(b.pipelineName)}} · ${{escapeHtml(b.stepTitle)}}</div>
      ${{here ? '' : '<div style="color:#888;font-style:italic">not in this version of the report</div>'}}
      <button class="bm-del" data-k="${{k}}" title="Remove">✕</button>
    </div>`;
  }}).join('');
  el.querySelectorAll('.bookmark-item').forEach(node => {{
    node.addEventListener('click', (e) => {{
      if (e.target.classList.contains('bm-del')) return;
      const b = bookmarks[parseInt(node.dataset.k)];
      const here = allItems().some(x => x.p.key === b.pipeline && x.s.id === b.step && x.i.file === b.file);
      if (here) goTo(b.pipeline, b.step, b.file);
    }});
  }});
  el.querySelectorAll('.bm-del').forEach(btn => {{
    btn.addEventListener('click', (e) => {{
      e.stopPropagation();
      const b = bookmarks[parseInt(btn.dataset.k)];
      if (!confirm(`Delete the bookmark "${{b.title}}"?`)) return;
      bookmarks.splice(parseInt(btn.dataset.k), 1);
      saveBookmarks();
      renderBookmarks();
      updateBookmarkButton();
    }});
  }});
}}

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

// Escapes first, then allows **bold** -- so figure-description text can
// highlight a word without any raw HTML in figure_descriptions.py.
function richText(s) {{
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
}}

// Renders an about.items list ([[label, text_or_bullets], ...]) as a
// <dt>/<dd> definition list -- a bullet list under a label when the value
// is an array, inline text otherwise.
function aboutRows(items) {{
  return (items || []).map(([label, text]) => {{
    const body = Array.isArray(text)
      ? `<ul>${{text.map(t => `<li>${{richText(t)}}</li>`).join('')}}</ul>`
      : richText(text);
    return `<dt>${{escapeHtml(label)}}</dt><dd>${{body}}</dd>`;
  }}).join('');
}}

async function copyText(text) {{
  try {{ await navigator.clipboard.writeText(text); return true; }} catch (e) {{ /* file:// fallback below */ }}
  const ta = document.createElement('textarea');
  ta.style.position = 'fixed'; ta.style.left = '-9999px'; ta.value = text;
  document.body.appendChild(ta); ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  return ok;
}}
function downloadText(text, name) {{
  const url = URL.createObjectURL(new Blob(['﻿' + text], {{type: 'text/csv;charset=utf-8'}}));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}}

// --- Prism tables (prism_tables.py): rows are a plain array of arrays, not
// the {{columns, rows}} object-row shape a stats table uses above. ---
function prismRowsText(rows, sep) {{
  const cell = v => {{
    const t = (v === null || v === undefined) ? '' : String(v);
    return sep === ',' && /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  }};
  return rows.map(r => r.map(cell).join(sep)).join(sep === ',' ? '\r\n' : '\n');
}}
// The whole table for one paste at Prism's top-left corner cell: the data
// set titles row (rows[0]) on top, then the value rows.
function prismTableBlock(rows, p) {{
  return [rows[0], ...rows.slice(p.r, p.n === null ? undefined : p.r + p.n)];
}}
// Every data set's session-ID subcolumn titles, one per line, each data set
// padded to the same count so a paste that runs into the next data set
// stays aligned.
function prismIdsText(subcolumns) {{
  const n = Math.max(...subcolumns.map(([, titles]) => titles.length));
  return subcolumns.flatMap(([, titles]) => [...titles, ...Array(n - titles.length).fill('')]).join('\n');
}}
function flashPrism(btn, text) {{
  const status = btn.parentElement.querySelector('.prism-status');
  if (!status) return;
  status.textContent = text;
  clearTimeout(status._t);
  status._t = setTimeout(() => {{ status.textContent = ''; }}, 2500);
}}
// A Prism card's numbered how-to-paste steps (Table/Mouse IDs/Data/Graph/Analysis).
function prismStepsHtml(steps) {{
  return steps.map(([label, items], i) => items.length === 1
    ? `<div class="prism-step">${{i + 1}}. <b>${{escapeHtml(label)}}:</b> ${{richText(items[0])}}</div>`
    : `<div class="prism-step">${{i + 1}}. <b>${{escapeHtml(label)}}:</b></div><ol class="prism-sub">${{items.map(x => `<li>${{richText(x)}}</li>`).join('')}}</ol>`
  ).join('');
}}
// The pipeline figure's title and axis titles (stored in the PDF when it
// was saved by figure_labels.py), to type into Prism's graph.
function figureTitlesHtml(labels) {{
  if (!labels) return '<div class="prism-line">Not stored in this figure -- rerun the notebook to list them here.</div>';
  const line = (name, text) => text ? `<div class="prism-line"><span class="prism-key">${{escapeHtml(name)}}: </span>${{escapeHtml(text)}}</div>` : '';
  const panels = labels.panels || [];
  const many = panels.filter(p => !p.colorbar).length > 1;
  const shown = panels.slice(0, 12);
  let n = 0;
  const panelsHtml = shown.map(p => p.colorbar ? line('Colour bar title', p.colorbar) : [
    many ? `<div class="prism-line prism-panel">Panel ${{++n}}</div>` : '',
    line((many ? '  ' : '') + (labels.title && !many ? 'Panel title' : 'Title'), p.title),
    line((many ? '  ' : '') + 'X title', p.x || '(none)'),
    line((many ? '  ' : '') + 'Y title', p.y || '(none)'),
  ].join('')).join('');
  return `<div class="prism-titles"><b>Titles in the pipeline's figure:</b></div>`
    + line('Title', labels.title) + panelsHtml
    + (panels.length > shown.length ? `<div class="prism-line">... ${{panels.length - shown.length}} more panels</div>` : '')
    + (!labels.title && !panels.length ? '<div class="prism-line">(the figure has no titles)</div>' : '');
}}
// The Prism tables for the step's current Comparison + Zone set selection,
// and (when that selection's tables are split by region, e.g. the zones of
// an "all regions" overlay) the regions to pick from inside the card.
function prismZoneSet(s) {{
  // Fall back to tables with no zone set (a datasheet written before figures
  // were tagged) rather than hide the Prism card.
  const cmp = state.comparison || s.defaultComparison;
  const has = zs => Object.keys(s.prism.tables).some(k => k.startsWith(`${{cmp}}|${{zs}}|`));
  return state.zoneSet && has(state.zoneSet) ? state.zoneSet : '_';
}}
function prismRegions(s) {{
  return (s.prism.regions || {{}})[prismZoneSet(s)] || null;
}}
function prismKey(s) {{
  const cmp = state.comparison || s.defaultComparison;
  const regions = prismRegions(s);
  const region = regions ? (regions.includes(state.prismRegion) ? state.prismRegion : regions[0]) : '_';
  return `${{cmp}}|${{prismZoneSet(s)}}|${{region}}`;
}}
function prismBlockHtml(s, labels) {{
  const t = s.prism.tables[prismKey(s)];
  if (!t) return '';
  const regions = prismRegions(s);
  const current = prismKey(s).split('|')[2];
  const regionSelect = regions
    ? `<label class="prism-status">Region <select data-action="prism-region">${{regions.map(r =>
        `<option value="${{escapeHtml(r)}}" ${{r === current ? 'selected' : ''}}>${{escapeHtml(r)}}</option>`).join('')}}</select></label>`
    : '';
  return `<details class="dd prism" ${{state.prismOpen ? 'open' : ''}}>
    <summary>Prism — copy the table, or download it as CSV</summary>
    <div class="dd-body">
      <div class="prism-row">
        ${{regionSelect}}
        <button class="pill" data-action="prism-copy" title="Titles and values in one block: in Prism, click the corner cell with the gray triangle (above the row numbers) and paste">⧉ Copy table</button>
        ${{t.subcolumns ? '<button class="pill" data-action="prism-copy-ids">⧉ Copy session IDs</button>' : ''}}
        <button class="pill" data-action="prism-download">⤓ CSV for Prism</button>
        <span class="prism-status"></span>
      </div>
      <div class="prism-about"><b>Values:</b> ${{richText(t.steps.about)}}</div>
      ${{t.steps.layout ? `<div class="prism-about"><b>Layout:</b> ${{richText(t.steps.layout)}}</div>` : ''}}
      ${{prismStepsHtml(t.steps.steps)}}
      ${{figureTitlesHtml(labels)}}
      <div class="prism-foot">${{richText(t.steps.note)}}</div>
    </div>
  </details>`;
}}
function renderTabs() {{
  const el = document.getElementById('tabs');
  const mazeBtn = DATA.mazeSvg ? `<button class="tab-mini ${{state.mode === 'maze' ? 'active' : ''}}" data-maze="1">Maze layout</button>` : '';
  el.innerHTML = mazeBtn + DATA.pipelines.map(p =>
    `<button class="tab-btn ${{state.mode === 'step' && p.key === state.pipeline ? 'active' : ''}}" data-key="${{p.key}}">${{escapeHtml(p.name)}} (${{p.figureCount}})</button>`
  ).join('');
  el.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {{
    state.mode = 'step';
    state.pipeline = btn.dataset.key;
    const p = DATA.pipelines.find(p => p.key === state.pipeline);
    const remembered = state.perPipeline[state.pipeline];
    if (remembered && p.steps.some(s => s.id === remembered.step)) {{
      state.step = remembered.step;
      state.comparison = remembered.comparison;
    }} else {{
      // First time this tab is opened this session -- default straight to
      // its first step instead of an empty "pick a step" placeholder.
      state.step = p.steps.length ? p.steps[0].id : null;
      state.comparison = null;
    }}
    saveViewState();
    renderAll();
  }}));
  const mazeNode = el.querySelector('[data-maze]');
  if (mazeNode) mazeNode.addEventListener('click', () => {{
    if (state.mode !== 'maze') mazeReturn = currentViewTitle();
    state.mode = 'maze';
    saveViewState();
    renderTabs();
    renderMain();
  }});
}}

let shownDatasheetPath = '';  // toggled by the "show datasheet location" button below

async function showDatasheetLocation(relPath) {{
  if (shownDatasheetPath) {{ shownDatasheetPath = ''; renderLeftbar(); return; }}
  let path = relPath;
  if (location.protocol === 'file:') {{
    const page = decodeURIComponent(location.pathname);
    const dir = page.slice(0, page.lastIndexOf('/') + 1) + relPath;
    path = /^\/[A-Za-z]:/.test(dir) ? dir.slice(1).split('/').join(String.fromCharCode(92)) : dir;
  }}
  const copied = await copyText(path);
  shownDatasheetPath = path;
  renderLeftbar();
}}

function renderLeftbar() {{
  const el = document.getElementById('leftbar');
  if (state.query.trim()) {{ renderSearchResults(el); return; }}
  const nSig = DATA.significant.length;
  const p = DATA.pipelines.find(p => p.key === state.pipeline);
  let html = `<button class="nav-fixed-item" data-action="significant">★ Significant results (${{nSig}})</button>`;
  if (p.datasheet) {{
    html += `<a class="nav-fixed-item" href="${{encodeURI(p.datasheet)}}" download style="text-decoration:none;display:block">⤓ Download datasheet (Excel)</a>`;
    html += `<button class="nav-fixed-item" data-action="datasheet-location" title="Shows the datasheet folder path and copies it, to paste into File Explorer / Finder">📁 Show datasheet location</button>`;
    if (shownDatasheetPath) {{
      html += `<div class="pathbox"><button class="pathclose" title="Hide" data-action="datasheet-location-close">✕</button><span>Copied to clipboard:</span><code>${{escapeHtml(shownDatasheetPath)}}</code></div>`;
    }}
  }}
  html += '<div class="nav-group-title">STEPS</div>';
  html += p.steps.map(s => `<button class="nav-step ${{s.id === state.step ? 'active' : ''}}" data-id="${{s.id}}">${{escapeHtml(s.title)}} <span style="color:#999">(${{s.items.length}})</span></button>`).join('');
  el.innerHTML = html;
  const dsLocBtn = el.querySelector('[data-action="datasheet-location"]');
  if (dsLocBtn) dsLocBtn.addEventListener('click', () => showDatasheetLocation(p.datasheet));
  const dsLocClose = el.querySelector('[data-action="datasheet-location-close"]');
  if (dsLocClose) dsLocClose.addEventListener('click', () => {{ shownDatasheetPath = ''; renderLeftbar(); }});
  el.querySelectorAll('.nav-step').forEach(btn => btn.addEventListener('click', () => {{
    state.mode = 'step';
    state.step = btn.dataset.id;
    saveViewState();
    renderTabs();
    renderMain();
    renderLeftbar();
  }}));
  const sigBtn = el.querySelector('[data-action="significant"]');
  if (sigBtn) sigBtn.addEventListener('click', () => {{ state.mode = 'significant'; state.step = null; renderTabs(); renderMain(); }});
}}

function renderSearchResults(el) {{
  const q = state.query.trim().toLowerCase();
  let html = '';
  DATA.pipelines.forEach(p => {{
    const matches = p.steps.filter(s => s.title.toLowerCase().includes(q) || s.items.some(i => i.title.toLowerCase().includes(q)));
    if (!matches.length) return;
    html += `<div class="search-group"><h4>${{escapeHtml(p.name)}}</h4>` +
      matches.map(s => `<button class="nav-step" data-p="${{p.key}}" data-id="${{s.id}}">${{escapeHtml(s.title)}}</button>`).join('') +
      '</div>';
  }});
  el.innerHTML = html || '<p style="color:#888;font-size:13px">No matches.</p>';
  el.querySelectorAll('.nav-step').forEach(btn => btn.addEventListener('click', () => {{
    state.mode = 'step';
    state.pipeline = btn.dataset.p;
    state.step = btn.dataset.id;
    state.query = '';
    document.getElementById('search').value = '';
    saveViewState();
    renderAll();
  }}));
}}

// --- Browser Back/Forward: every view is a URL hash -------------------------
// m=maze | m=sig | m=zones | p=<pipeline>&s=<step>&c=<comparison>&z=<zone set>.
// A different page pushes a history entry; changing only the comparison /
// zone set on the same page replaces it (so Back doesn't replay pill clicks).
let historyReady = false;
function viewHash() {{
  if (state.mode === 'maze') return 'm=maze';
  if (state.mode === 'significant') return 'm=sig';
  if (state.mode === 'zonesets') return 'm=zones';
  const q = new URLSearchParams({{p: state.pipeline || '', s: state.step || ''}});
  if (state.comparison) q.set('c', state.comparison);
  if (state.zoneSet) q.set('z', state.zoneSet);
  const pl = DATA.pipelines.find(p => p.key === state.pipeline);
  const step = pl && pl.steps.find(s => s.id === state.step);
  (step ? step.selectors : []).forEach(sel => {{
    if (state.options[sel.key]) q.set('o.' + sel.key, state.options[sel.key]);
  }});
  return q.toString();
}}
const pageOf = h => {{ const q = new URLSearchParams(h.replace(/^#/, '')); return (q.get('m') || '') + '|' + (q.get('p') || '') + '|' + (q.get('s') || ''); }};
// Plain hash navigation (not history.pushState, which file:// pages may
// refuse): setting location.hash adds an entry, location.replace() doesn't.
const scrollByHash = {{}};
function syncHistory() {{
  const hash = '#' + viewHash();
  if (hash === location.hash) return;
  if (!historyReady || pageOf(hash) === pageOf(location.hash)) location.replace(hash);
  else location.hash = hash;
}}
function applyHash(h) {{
  const q = new URLSearchParams(h.replace(/^#/, ''));
  const m = q.get('m');
  if (m === 'maze') state.mode = 'maze';
  else if (m === 'sig') {{ state.mode = 'significant'; state.step = null; }}
  else if (m === 'zones') state.mode = 'zonesets';
  else if (q.get('p') && q.get('s')) {{
    state.mode = 'step';
    state.pipeline = q.get('p');
    state.step = q.get('s');
    if (q.get('c')) state.comparison = q.get('c');
    if (q.get('z')) state.zoneSet = q.get('z');
    for (const [k, v] of q) if (k.startsWith('o.')) state.options[k.slice(2)] = v;
  }} else return false;
  return true;
}}
window.addEventListener('hashchange', () => {{
  // Already showing this view (the page set the hash itself) -- nothing to do
  if (location.hash === '#' + viewHash()) return;
  if (!applyHash(location.hash)) return;
  saveViewState();
  renderAll();
  document.getElementById('main').scrollTop = scrollByHash[location.hash] || 0;
}});
// Remember each view's scroll position, for coming back to it
document.getElementById('main').addEventListener('scroll', () => {{
  scrollByHash[location.hash] = document.getElementById('main').scrollTop;
}}, {{passive: true}});

function renderMain() {{
  renderView();
  syncHistory();
}}

function renderView() {{
  const el = document.getElementById('main');
  if (state.mode === 'significant') {{ renderSignificantPage(); return; }}
  if (state.mode === 'maze') {{
    const back = mazeReturn ? `<button class="back-btn" data-action="maze-back">← Back to ${{escapeHtml(mazeReturn)}}</button>` : '';
    el.innerHTML = `${{back}}<h1 class="step-heading">Maze layout</h1><div class="step-count">Zone/cell-number reference for the 10x10 hairpin maze</div><div class="maze-wrap">${{DATA.mazeSvg}}</div>`;
    const backBtn = el.querySelector('[data-action="maze-back"]');
    if (backBtn) backBtn.addEventListener('click', () => history.back());
    updateBookmarkButton();
    return;
  }}
  if (state.mode === 'zonesets') {{ renderZoneSetsPage(el); return; }}
  if (!state.step) {{ el.innerHTML = '<p style="color:#888">Pick a step from the left.</p>'; updateBookmarkButton(); return; }}
  const p = DATA.pipelines.find(p => p.key === state.pipeline);
  const s = p.steps.find(s => s.id === state.step);
  if (!s) {{ el.innerHTML = ''; return; }}

  // One genotype-comparison selector and one zone-set selector per step,
  // listing only what this step's figures were actually made with (read off
  // their file names). They choose the figures, the statistics tables and
  // the Prism table together. Reset to the first option whenever the step
  // changes or the previous pick doesn't apply here.
  if (!s.comparisons.some(c => c.key === state.comparison)) {{
    state.comparison = s.comparisons.length ? s.comparisons[0].key : null;
  }}
  if (!s.zoneSets.includes(state.zoneSet)) {{
    state.zoneSet = s.zoneSets.length ? s.zoneSets[0] : null;
  }}
  // A step's own extra selectors (e.g. Plot 7's direction, time bins)
  s.selectors.forEach(sel => {{
    if (!sel.options.some(o => o.key === state.options[sel.key])) {{
      state.options[sel.key] = sel.options.length ? sel.options[0].key : null;
    }}
  }});
  // Remember this step+comparison as this pipeline's "last viewed", so
  // switching tabs and coming back restores it -- covers every way of
  // landing on a step (direct nav click, search, bookmarks, significant
  // results, the comparison pills), not just the tab click itself.
  state.perPipeline[state.pipeline] = {{step: state.step, comparison: state.comparison}};
  saveViewState();
  // A figure/table naming no comparison (or no zone set) shows under every selection.
  const matchesFor = (x, zoneSet) => (!state.comparison || !x.comparison || x.comparison === state.comparison)
    && (!zoneSet || !x.zoneSet || x.zoneSet === zoneSet)
    && s.selectors.every(sel => {{
      const want = state.options[sel.key], have = (x.options || {{}})[sel.key];
      return !want || !have || have === want;
    }});
  // Zone sets with a figure for the other selections (e.g. a zone set made for
  // one direction only); the rest are greyed out. If the current one has none
  // (a selector just changed), fall back to the first that does.
  const zoneSetOk = z => s.items.some(x => matchesFor(x, z));
  if (state.zoneSet && !zoneSetOk(state.zoneSet)) {{
    const ok = s.zoneSets.find(zoneSetOk);
    if (ok) state.zoneSet = ok;
  }}
  const matches = x => matchesFor(x, state.zoneSet);
  const visibleItems = s.items.filter(matches);
  const visibleStats = s.stats.filter(matches);

  let html = '<div class="controls"><h1 class="step-heading">' + escapeHtml(s.title) + '</h1><div class="step-count">'
    + visibleItems.length + (visibleItems.length !== s.items.length ? ' of ' + s.items.length : '') + ' figure(s)</div>';
  // Both rows on every step, listing only what this step has -- a single
  // option is shown selected; none at all reads "none".
  const none = '<span class="cmp-none">none</span>';
  html += '<div class="cmp-row"><span class="cmp-label">Comparison:</span>' + (s.comparisons.length ? s.comparisons.map(c =>
    `<button class="pill ${{c.key === state.comparison ? 'active' : ''}}" data-cmp="${{c.key}}">${{escapeHtml(c.label)}}</button>`
  ).join('') : none) + '</div>';
  html += '<div class="cmp-row"><a href="#" class="cmp-label cmp-link" data-action="zonesets" title="What each zone set contains">Zone set ⓘ</a>' + (s.zoneSets.length ? s.zoneSets.map(z =>
    `<button class="pill ${{z === state.zoneSet ? 'active' : ''}}" data-zoneset="${{escapeHtml(z)}}"${{zoneSetOk(z) ? '' : ' disabled title="Not available for the current selection"'}}>${{escapeHtml(z)}}</button>`
  ).join('') : none) + '</div>';
  s.selectors.forEach(sel => {{
    html += `<div class="cmp-row"><span class="cmp-label">${{escapeHtml(sel.label)}}:</span>` + (sel.options.length ? sel.options.map(o =>
      `<button class="pill ${{o.key === state.options[sel.key] ? 'active' : ''}}" data-sel="${{escapeHtml(sel.key)}}" data-option="${{escapeHtml(o.key)}}">${{escapeHtml(o.label)}}</button>`
    ).join('') : none) + '</div>';
  }});
  html += '</div>';
  if (!visibleItems.length) {{
    html += '<p style="color:#888">No figures generated yet for this step -- rerun the notebook cell to populate it.</p>';
  }}
  visibleItems.forEach((item, idx) => {{
    const cardId = `card-${{p.key}}-${{s.id}}-${{idx}}`;
    html += `<div class="card" id="${{cardId}}" data-file="${{escapeHtml(item.file)}}" data-title="${{escapeHtml(item.title)}}">
      <div class="card-title">
        <b>${{escapeHtml(item.title)}}</b>
        <button data-action="actual-size">Actual size</button>
        <a href="figures/${{p.key}}/${{s.id}}/${{encodeURIComponent(item.file)}}" target="_blank">Open file</a>
      </div>
      <a href="figures/${{p.key}}/${{s.id}}/${{encodeURIComponent(item.file)}}" target="_blank">
        <img ${{item.fit ? 'class="fit" ' : ''}}${{item.width ? `data-w="1" style="--w:${{item.width}}%" ` : ''}}src="${{imgSrc(item.image)}}" width="${{item.size[0]}}" height="${{item.size[1]}}" loading="lazy">
      </a>
      <details class="dd"><summary>Figure description — <span class="lead">${{richText(s.about.summary)}}</span></summary><div class="dd-body about-body"><dl>${{aboutRows(s.about.items)}}</dl></div></details>`;
    if (visibleStats.length) {{
      const nSig = visibleStats.reduce((n, t) => n + (t.sigCol ? t.rows.filter(r => typeof r[t.sigCol] === 'number' && r[t.sigCol] < 0.05).length : 0), 0);
      html += `<details class="dd"><summary>Statistics — ${{visibleStats.reduce((n, t) => n + t.rows.length, 0)}} rows, <b>${{nSig}} significant</b></summary><div class="dd-body">`;
      visibleStats.forEach((t, ti) => {{
        html += `<div style="margin:6px 0 2px"><b>${{escapeHtml(t.name)}}</b></div>`;
        if (t.sigCol) html += `<label class="toggle-sig"><input type="checkbox" class="sig-only" data-ti="${{ti}}" data-cardidx="${{idx}}"> significant rows only (${{t.sigCol}} &lt; 0.05)</label>`;
        html += renderStatsTable(t, false, idx, ti);
      }});
      html += '</div></details>';
    }}
    if (s.prism) html += prismBlockHtml(s, item.labels);
    html += '</div>';
  }});
  el.innerHTML = html;
  updateFitHeight();
  el.querySelectorAll('[data-action="actual-size"]').forEach(btn => btn.addEventListener('click', () => {{
    btn.closest('.card').classList.toggle('actual-size');
  }}));
  el.querySelectorAll('.sig-only').forEach(cb => cb.addEventListener('change', () => {{
    const t = visibleStats[parseInt(cb.dataset.ti)];
    rerenderStatsTable(cb, t);
  }}));
  el.querySelectorAll('[data-cmp]').forEach(btn => btn.addEventListener('click', () => {{
    state.comparison = btn.dataset.cmp;
    renderMain();
  }}));
  el.querySelectorAll('[data-zoneset]').forEach(btn => btn.addEventListener('click', () => {{
    state.zoneSet = btn.dataset.zoneset;
    renderMain();
  }}));
  el.querySelectorAll('[data-option]').forEach(btn => btn.addEventListener('click', () => {{
    state.options[btn.dataset.sel] = btn.dataset.option;
    renderMain();
  }}));
  const zsLink = el.querySelector('[data-action="zonesets"]');
  if (zsLink) zsLink.addEventListener('click', e => {{
    e.preventDefault();
    // Where Back returns to: this step, its selections and scroll position
    zoneSetsReturn = {{pipeline: state.pipeline, step: state.step, comparison: state.comparison,
                       zoneSet: state.zoneSet, scroll: el.scrollTop}};
    state.mode = 'zonesets';
    renderTabs();
    renderMain();
    el.scrollTop = 0;
  }});
  if (s.prism) {{
    const key = prismKey(s);
    el.querySelectorAll('[data-action="prism-region"]').forEach(sel => sel.addEventListener('change', () => {{
      state.prismRegion = sel.value;
      state.prismOpen = true;  // keep the card open across the re-render
      renderMain();
      state.prismOpen = false;
    }}));
    el.querySelectorAll('[data-action="prism-copy"]').forEach(btn => btn.addEventListener('click', async () => {{
      const t = s.prism.tables[key];
      const ok = await copyText(prismRowsText(prismTableBlock(t.rows, t.paste), '\t'));
      flashPrism(btn, ok ? 'table copied' : 'copy failed, use the CSV');
    }}));
    el.querySelectorAll('[data-action="prism-copy-ids"]').forEach(btn => btn.addEventListener('click', async () => {{
      const t = s.prism.tables[key];
      const ok = await copyText(prismIdsText(t.subcolumns));
      flashPrism(btn, ok ? 'session IDs copied' : 'copy failed');
    }}));
    el.querySelectorAll('[data-action="prism-download"]').forEach(btn => btn.addEventListener('click', () => {{
      const t = s.prism.tables[key];
      const name = `${{s.id}}_${{key.split('|').filter(k => k !== '_').join('_')}}_prism.csv`;
      downloadText(prismRowsText(t.rows, ','), name);
      flashPrism(btn, 'saved ' + name);
    }}));
  }}
  updateBookmarkButton();
}}

function renderStatsTable(t, sigOnly, cardIdx, ti) {{
  let rows = t.rows;
  if (sigOnly && t.sigCol) rows = rows.filter(r => typeof r[t.sigCol] === 'number' && r[t.sigCol] < 0.05);
  let h = `<table class="stats" data-cardidx="${{cardIdx}}" data-ti="${{ti}}"><thead><tr>` + t.columns.map(c => `<th>${{escapeHtml(c)}}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(r => {{
    const isSig = t.sigCol && typeof r[t.sigCol] === 'number' && r[t.sigCol] < 0.05;
    h += `<tr class="${{isSig ? 'sig' : ''}}">` + t.columns.map(c => {{
      let v = r[c];
      if (typeof v === 'number') v = Number.isInteger(v) ? v : v.toFixed(4);
      return `<td>${{escapeHtml(v === null || v === undefined ? '' : v)}}</td>`;
    }}).join('') + '</tr>';
  }});
  h += '</tbody></table>';
  return h;
}}
function rerenderStatsTable(cb, t) {{
  const body = cb.closest('.dd-body');
  const oldTable = body.querySelector(`table.stats[data-cardidx="${{cb.dataset.cardidx}}"][data-ti="${{cb.dataset.ti}}"]`);
  if (!oldTable) return;
  oldTable.outerHTML = renderStatsTable(t, cb.checked, cb.dataset.cardidx, cb.dataset.ti);
}}

// --- Zone sets reference page (opened from a step's "Zone set" link) ---
let zoneSetsReturn = null;
let mazeReturn = null;  // title of the view the Maze layout tab was opened from (for its Back button)

function currentViewTitle() {{
  if (state.mode === 'significant') return 'Significant results';
  if (state.mode === 'zonesets') return 'Zone sets';
  const p = DATA.pipelines.find(p => p.key === state.pipeline);
  const s = p && p.steps.find(s => s.id === state.step);
  return s ? s.title : 'previous page';
}}

// A zone set / zone name that copies itself in quotes ("name"), ready to paste
// into a notebook's ZONE_SET = [...] list
const copyName = name => `<button class="zs-name" data-copy-name="${{escapeHtml(name)}}" title="Copy &quot;${{escapeHtml(name)}}&quot; to paste into ZONE_SET">"${{escapeHtml(name)}}"<span class="zs-copy-hint">copy</span></button>`;

function renderZoneSetsPage(el) {{
  const info = DATA.zoneInfo || {{sets: {{}}, zones: {{}}, usedBy: {{}}}};
  const current = zoneSetsReturn ? zoneSetsReturn.zoneSet : null;
  const cells = z => (info.zones[z] || []).join(', ') || '<span class="cmp-none">not defined</span>';
  const backLabel = zoneSetsReturn
    ? (DATA.pipelines.find(p => p.key === zoneSetsReturn.pipeline)?.steps.find(s => s.id === zoneSetsReturn.step)?.title || 'previous page')
    : 'previous page';
  let html = `<div class="controls"><button class="back-btn" data-action="zonesets-back">← Back to ${{escapeHtml(backLabel)}}</button>
    <h1 class="step-heading">Zone sets</h1>
    <div class="step-count">Each figure's zones come from one zone set in config.yaml; numbers are maze grid cells (see the Maze layout tab).</div></div>`;
  const names = Object.keys(info.sets);
  if (!names.length) html += '<p style="color:#888">No zone sets defined in config.yaml.</p>';
  names.forEach(name => {{
    const used = info.usedBy[name] || [];
    html += `<div class="zs-card${{name === current ? ' zs-current' : ''}}" id="zs-${{escapeHtml(name)}}">
      <div class="zs-head">${{copyName(name)}}${{name === current ? ' <span class="zs-tag">selected</span>' : ''}}
        <span class="zs-meta">${{info.sets[name].length}} zone(s)</span></div>
      <table class="zs-table"><tr><th>Zone</th><th>Grid cells</th></tr>` +
      info.sets[name].map(z => `<tr><td>${{escapeHtml(z)}}</td><td>${{cells(z)}}</td></tr>`).join('') +
      `</table><div class="zs-used">${{used.length ? 'Used by: ' + used.map(escapeHtml).join('; ') : 'Not used by any figure yet.'}}</div></div>`;
  }});
  // Single zones a step uses directly instead of a set (e.g. Plot 4's top_corners)
  const single = Object.keys(info.usedBy).filter(z => !(z in info.sets));
  if (single.length) {{
    html += '<h3 style="margin:22px 0 8px">Single zones used directly</h3>';
    single.forEach(z => {{
      html += `<div class="zs-card${{z === current ? ' zs-current' : ''}}" id="zs-${{escapeHtml(z)}}"><div class="zs-head">${{copyName(z)}}${{z === current ? ' <span class="zs-tag">selected</span>' : ''}}</div>
        <table class="zs-table"><tr><th>Grid cells</th></tr><tr><td>${{cells(z)}}</td></tr></table>
        <div class="zs-used">Used by: ${{info.usedBy[z].map(escapeHtml).join('; ')}}</div></div>`;
    }});
  }}
  el.innerHTML = html;
  updateBookmarkButton();
  el.querySelectorAll('[data-copy-name]').forEach(btn => btn.addEventListener('click', async () => {{
    await copyText(`"${{btn.dataset.copyName}}"`);
    btn.classList.add('copied');
    btn.querySelector('.zs-copy-hint').textContent = 'copied';
    setTimeout(() => {{ btn.classList.remove('copied'); btn.querySelector('.zs-copy-hint').textContent = 'copy'; }}, 1200);
  }}));
  el.querySelector('[data-action="zonesets-back"]').addEventListener('click', () => {{
    const r = zoneSetsReturn;
    // Opened from a step's link: the step is the previous history entry, so
    // go back to it (same as the browser's Back button)
    if (r) {{ history.back(); return; }}
    state.mode = 'step';
    if (r) {{ state.pipeline = r.pipeline; state.step = r.step; state.comparison = r.comparison; state.zoneSet = r.zoneSet; }}
    zoneSetsReturn = null;
    saveViewState();
    renderAll();
    if (r) el.scrollTop = r.scroll;
  }});
  if (current) {{
    const card = document.getElementById('zs-' + current);
    if (card) card.scrollIntoView({{block: 'center'}});
  }}
}}

// A "fit" figure (small, e.g. a bar graph) is sized to the window height
// minus its card's title row and a small margin, so once the controls are
// scrolled past, the whole figure and its title are visible together.
function updateFitHeight() {{
  const main = document.getElementById('main');
  const title = main.querySelector('.card-title');
  const used = (title ? title.offsetHeight : 40) + 40;
  const avail = Math.max(300, main.clientHeight - used);
  document.documentElement.style.setProperty('--fit-h', avail + 'px');
}}
window.addEventListener('resize', () => requestAnimationFrame(updateFitHeight));

function renderSignificantPage() {{
  const el = document.getElementById('main');
  let html = `<h1 class="step-heading">Significant results</h1><div class="step-count">${{DATA.significant.length}} row(s) below 0.05</div>`;
  if (!DATA.significant.length) html += '<p style="color:#888">None found.</p>';
  DATA.significant.forEach(s => {{
    html += `<div class="sig-row-link" data-p="${{s.pipeline}}" data-s="${{s.step}}">
      <b>${{escapeHtml(s.pipelineName)}} · ${{escapeHtml(s.stepTitle)}}</b> -- ${{escapeHtml(s.table)}}:
      ${{escapeHtml(s.sigCol)}} = ${{Number(s.value).toFixed(4)}}
    </div>`;
  }});
  el.innerHTML = html;
  updateBookmarkButton();
  el.querySelectorAll('.sig-row-link').forEach(node => node.addEventListener('click', () => {{
    state.mode = 'step';
    state.pipeline = node.dataset.p;
    state.step = node.dataset.s;
    saveViewState();
    renderTabs();
    renderMain();
    renderLeftbar();
  }}));
}}

function goTo(pipelineKey, stepId, file) {{
  state.mode = 'step';
  state.pipeline = pipelineKey;
  state.step = stepId;
  // Make sure the target figure's own comparison is selected, or a filtered
  // view (e.g. "WT vs. NLF" selected while the bookmark is an "all 4" figure)
  // would hide it right after navigating to it.
  const p = DATA.pipelines.find(p => p.key === pipelineKey);
  const s = p && p.steps.find(s => s.id === stepId);
  const item = s && s.items.find(i => i.file === file);
  if (item && item.comparison) state.comparison = item.comparison;
  if (item && item.zoneSet) state.zoneSet = item.zoneSet;
  if (item && item.options) Object.entries(item.options).forEach(([k, v]) => {{ if (v) state.options[k] = v; }});
  saveViewState();
  renderTabs();
  renderLeftbar();
  renderMain();
  setTimeout(() => {{
    const card = document.querySelector(`.card[data-file="${{CSS.escape(file)}}"]`);
    if (card) {{
      card.scrollIntoView({{block: 'start'}});
      card.classList.add('highlight-flash');
      setTimeout(() => card.classList.remove('highlight-flash'), 1500);
    }}
  }}, 30);
}}

function renderAll() {{
  renderTabs();
  renderLeftbar();
  renderMain();
  renderBookmarks();
}}

document.getElementById('search').addEventListener('input', (e) => {{
  state.query = e.target.value;
  renderLeftbar();
}});
document.getElementById('bmbtn').addEventListener('click', addBookmark);
document.getElementById('main').addEventListener('scroll', () => requestAnimationFrame(updateBookmarkButton), {{passive: true}});

// A shared link (#p=...&s=...) always wins -- jump straight there. Otherwise
// sessionStorage restores exactly where a same-tab refresh left off; with
// neither, state's own default (mode: 'maze') is what a genuinely fresh
// open (new tab/window, or reopening the file later) starts on.
loadViewState();
if (state.mode === 'datasheets') state.mode = 'maze';  // page since removed
if (!applyHash(location.hash) && !DATA.pipelines.some(p => p.key === state.pipeline)) {{
  state.pipeline = DATA.pipelines[0].key;
}}
if (state.mode === 'step' && !state.step) {{
  // loadViewState() restored mode/pipeline/perPipeline but state.step itself
  // isn't persisted directly -- look it up the same way switching tabs does.
  const p = DATA.pipelines.find(p => p.key === state.pipeline);
  const remembered = state.perPipeline[state.pipeline];
  if (p && remembered && p.steps.some(s => s.id === remembered.step)) {{
    state.step = remembered.step;
    state.comparison = remembered.comparison;
  }} else if (p && p.steps.length) {{
    state.step = p.steps[0].id;
  }}
}}
if (!DATA.pipelines.some(p => p.key === state.pipeline)) state.pipeline = DATA.pipelines[0].key;

renderAll();
historyReady = true;
</script>
</body>
</html>
"""
