"""Each saved figure's title and axis titles, stored in the PDF's own
metadata (its Subject field) so the figure report can list them on the
figure's Prism card -- no extra files, no per-plot wiring.

install() wraps matplotlib's Figure.savefig once per process; every PDF any
plotting function below saves afterward carries its labels automatically.
read(path) returns what was stored, or None for a figure saved before this
was installed (e.g. before the notebook imports this module).
"""

import json
import os

from matplotlib.figure import Figure

KEY = "compass-labyrinth-figure-labels"


def _text(t):
    return " ".join(str(t).split())


def labels_of(fig):
    """{"title": suptitle or "", "panels": [{"title", "x", "y"}, ...]} --
    one entry per distinct panel labelling (colour bars give their label as
    "colorbar"); empty panels skipped."""
    suptitle = _text(fig._suptitle.get_text()) if getattr(fig, "_suptitle", None) else ""
    panels, seen = [], set()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if getattr(ax, "_colorbar", None) is not None:
            label = _text(ax.get_ylabel() or ax.get_xlabel())
            entry = {"colorbar": label} if label else None
        else:
            title = " / ".join(t for t in (_text(ax.get_title(loc)) for loc in ("left", "center", "right")) if t)
            entry = {k: v for k, v in (("title", title), ("x", _text(ax.get_xlabel())),
                                       ("y", _text(ax.get_ylabel()))) if v}
        key = json.dumps(entry, sort_keys=True)
        if entry and key not in seen:
            seen.add(key)
            panels.append(entry)
    return {"title": suptitle, "panels": panels}


def install():
    if getattr(Figure.savefig, "_figure_labels", False):
        return
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        path = os.fspath(fname) if isinstance(fname, (str, os.PathLike)) else ""
        fmt = kwargs.get("format") or os.path.splitext(path)[1].lstrip(".")
        if str(fmt).lower() == "pdf":
            try:
                meta = dict(kwargs.get("metadata") or {})
                meta.setdefault("Subject", KEY + json.dumps(labels_of(self)))
                kwargs["metadata"] = meta
            except Exception:
                pass
        return original(self, fname, *args, **kwargs)

    savefig._figure_labels = True
    Figure.savefig = savefig


def read(path):
    """The labels stored in a PDF saved after install(), else None."""
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            subject = (doc.metadata or {}).get("subject") or ""
    except Exception:
        return None
    return json.loads(subject[len(KEY):]) if subject.startswith(KEY) else None


install()
