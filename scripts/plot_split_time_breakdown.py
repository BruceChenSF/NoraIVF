#!/usr/bin/env python3
"""Split-time breakdown: vertical 100% stacked bars + average line.

Restores the *original* layout of the paper's split-time figure -- four
datasets as vertical 100% stacked columns, one dashed average line with a
small value box on the right, y axis 0-100% -- but feeds it the current
full-coverage claim-evidence data (no bounded-window runs, no log panel).

The four fixed streams (original figure) and their sources are::

    SIFT-1M      results/claim_evidence_sift.json['sift-128-euclidean']
    MNIST-60k    results/claim_evidence_small.json['mnist-784-euclidean']
    Fashion-60k  results/claim_evidence_small.json['fashion-mnist-784-euclidean']
    GIST-500k    results/claim_evidence_gist500k.json['gist-960-euclidean']

Each column splits SPFresh L=5 split-maintenance time into three parts that
sum to 100%::

    Boundary reassignment = methods.SPFresh.split_reassign_time_s
    Local k-means         = methods.SPFresh.split_clustering_time_s
    Other                 = methods.SPFresh.maintenance_time_s - the two above

Boundary reassignment is 99.4-99.8% of the total on every stream, so a plain
0-100% column is visually identical across datasets.  Rather than break the
axis (which would be dishonest) the figure keeps 0-100% and adds the absolute
reassignment time above each column; the two near-invisible slivers are
encoded in the legend instead of being crammed with text.  A dashed line marks
the arithmetic mean of the four reassignment shares and is labelled in a small
box on the right, matching the original figure.

Layout/type: native single-column canvas (3.34 in wide, <= 2.6 in high) so the
saved PDF/PNG already sit at the insertion size and every point size is the
effective on-page size (no compensation loop).  Effective sizes: axis title
7.25 pt, ticks/annotations 7.0 pt, legend 6.5 pt (>= 6.5 pt floor).  The figure
carries no multi-line explanatory text -- that belongs in the caption.

Outputs: figures/split_time_breakdown.pdf / .png
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figstyle import (  # noqa: E402
    SINGLE_COL_IN,
    report,
    save_native,
    scaled,
    scaled_rc,
)

_ROOT = Path(__file__).resolve().parent.parent
_FIGDIR = _ROOT / "figures"

# Effective type sizes for the native single-column canvas (canvas == insert).
SPLIT_TARGETS = {
    "tick": 7.0,     # tick labels            (spec 7.0-7.5)
    "label": 7.25,   # axis title             (spec 7.0-7.5)
    "text": 7.0,     # on-bar annotations
    "legend": 6.5,   # thin-segment legend    (hard floor)
    "title": 7.5,    # unused (single panel, no title)
}

# Three split-maintenance components (colours follow the paper figures:
# SPFresh blue for reassignment, neutral grey for the clustering sliver,
# mustard for the remaining accounted time).
REASSIGN_COLOR = "#1f77b4"
KMEANS_COLOR = "#9e9e9e"
OTHER_COLOR = "#e6ab02"

# (x label, claim-evidence file stem, dataset key) in the original figure's
# x order: SIFT-1M, MNIST-60k, Fashion-60k, GIST-500k.
DATASETS = [
    ("SIFT-1M", "sift", "sift-128-euclidean"),
    ("MNIST-60k", "small", "mnist-784-euclidean"),
    ("Fashion-60k", "small", "fashion-mnist-784-euclidean"),
    ("GIST-500k", "gist500k", "gist-960-euclidean"),
]

# Data contract: values from the current full-coverage claim-evidence JSONs.
# (reassign s, kmeans/local-clustering s, other s, (reassign %, kmeans %, other %))
EXPECTED = {
    "SIFT-1M": (1493.158, 1.950, 1.462, (99.77, 0.13, 0.10)),
    "MNIST-60k": (61.567, 0.251, 0.145, (99.36, 0.40, 0.23)),
    "Fashion-60k": (103.054, 0.363, 0.233, (99.42, 0.35, 0.23)),
    "GIST-500k": (2970.605, 3.501, 2.829, (99.79, 0.12, 0.10)),
}
_SEC_TOL = 2e-3   # JSON is rounded to 1 ms; expected values to 3 dp
_PCT_TOL = 1e-2


def fmt_seconds(seconds: float) -> str:
    """Absolute time with 3 significant figures ('1.49 ks', '61.6 s')."""
    if seconds >= 1000.0:
        return f"{seconds / 1e3:.2f} ks"
    if seconds >= 100.0:
        return f"{seconds:.0f} s"
    if seconds >= 10.0:
        return f"{seconds:.1f} s"
    return f"{seconds:.2f} s"


def load_data() -> list[dict]:
    rows = []
    for label, stem, key in DATASETS:
        path = _ROOT / "results" / f"claim_evidence_{stem}.json"
        with path.open(encoding="utf-8") as fh:
            rec = json.load(fh)[key]
        sp = rec["methods"]["SPFresh"]
        reassign = float(sp["split_reassign_time_s"])
        kmeans = float(sp["split_clustering_time_s"])
        total = float(sp["maintenance_time_s"])
        other = total - reassign - kmeans
        assert other >= 0.0, (label, "negative Other component", other)
        pct = (100.0 * reassign / total,
               100.0 * kmeans / total,
               100.0 * other / total)
        rows.append(
            {
                "label": label,
                "reassign": reassign,
                "kmeans": kmeans,
                "other": other,
                "total": total,
                "pct": pct,
            }
        )
    return rows


def _plot(rows: list[dict], scale: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
    d = scaled(SPLIT_TARGETS, scale)
    plt.rcParams.update(scaled_rc(scale, SPLIT_TARGETS))

    n = len(rows)
    x = np.arange(n)
    pct_r = np.array([r["pct"][0] for r in rows])
    pct_k = np.array([r["pct"][1] for r in rows])
    pct_o = np.array([r["pct"][2] for r in rows])
    width = 0.55

    # Native single-column canvas (width == insertion width).  Height 2.55 in
    # stays <= the 2.6 in single-column budget; the axes leave a small top
    # margin for the absolute-time labels and a right margin for the avg box.
    fig = plt.figure(figsize=(SINGLE_COL_IN, 2.55))
    ax = fig.add_axes([0.140, 0.205, 0.760, 0.695])

    ax.bar(x, pct_r, width, color=REASSIGN_COLOR, edgecolor="white",
           linewidth=0.5 * scale, zorder=3, label="Boundary reassignment")
    ax.bar(x, pct_k, width, bottom=pct_r, color=KMEANS_COLOR,
           edgecolor="white", linewidth=0.5 * scale, zorder=3,
           label="Local k-means")
    ax.bar(x, pct_o, width, bottom=pct_r + pct_k, color=OTHER_COLOR,
           edgecolor="white", linewidth=0.5 * scale, zorder=3, label="Other")

    # Per-column annotations: reassignment share in white inside the dominant
    # segment, absolute reassignment time above the column.
    for xi, row in zip(x, rows):
        ax.text(xi, row["pct"][0] / 2.0, f"{row['pct'][0]:.1f}%",
                ha="center", va="center", fontsize=d["text"], color="white",
                zorder=5)
        ax.annotate(fmt_seconds(row["reassign"]), xy=(xi, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 3.0 * scale), textcoords="offset points",
                    ha="center", va="bottom", fontsize=d["text"],
                    color="0.10", annotation_clip=False)

    # Dashed average-reassignment line + small boxed value on the right.
    avg = float(np.mean(pct_r))
    ax.axhline(avg, color="0.15", linestyle=(0, (4, 2)),
               linewidth=0.9 * scale, zorder=4)
    ax.text(1.0, avg, f"{avg:.1f}%", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=d["text"], color="0.10",
            clip_on=False, zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                      edgecolor="0.15", linewidth=0.6 * scale))

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100%"])
    ax.set_ylabel("Percentage of total split time", fontsize=d["label"],
                  labelpad=1.5 * scale)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=d["tick"])
    ax.tick_params(axis="both", which="major", pad=1.6 * scale,
                   length=2.6 * scale, labelsize=d["tick"])
    ax.grid(True, axis="y", linewidth=0.3 * scale, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Thin k-means/Other slivers are legend-encoded (never crammed on-bar).
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               ncol=2, frameon=False, fontsize=d["legend"], handlelength=1.1,
               handletextpad=0.4, columnspacing=0.9, borderaxespad=0.0)
    return fig, ax


def validate(rows: list[dict], ax: plt.Axes) -> None:
    """Assert the plotted data against the hard-coded data contract."""
    seg = [np.array([p.get_height() for p in c.patches])
           for c in ax.containers[:3]]
    for i, row in enumerate(rows):
        label = row["label"]
        exp_r, exp_k, exp_o, exp_pct = EXPECTED[label]
        assert abs(row["reassign"] - exp_r) < _SEC_TOL, (label, "reassign", row["reassign"])
        assert abs(row["kmeans"] - exp_k) < _SEC_TOL, (label, "kmeans", row["kmeans"])
        assert abs(row["other"] - exp_o) < _SEC_TOL, (label, "other", row["other"])
        for got, want, name in zip(row["pct"], exp_pct, ("reassign", "kmeans", "other")):
            assert abs(got - want) < _PCT_TOL, (label, name, "pct", got, want)
        # three components must sum to exactly 100%
        assert abs(sum(row["pct"]) - 100.0) < 1e-9, (label, "components != 100%", row["pct"])
        # the plotted geometry must match the percentages
        for j, (got, want) in enumerate(zip((seg[0][i], seg[1][i], seg[2][i]),
                                            row["pct"])):
            assert abs(got - want) < 1e-6, (label, "bar height", j, got, want)

    avg = float(np.mean([r["pct"][0] for r in rows]))
    assert abs(avg - 99.585) < 0.05, ("avg", avg)
    print(f"assertions OK: 4 datasets match expected s and %; "
          f"each column's 3 components sum to 100%; avg reassign = {avg:.3f}%")


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        fh.read(8)
        fh.read(4)
        assert fh.read(4) == b"IHDR"
        width, height = struct.unpack(">II", fh.read(8))
    return width, height


def main() -> None:
    rows = load_data()

    def build(scale: float) -> plt.Figure:
        fig, _ = _plot(rows, scale)
        return fig

    _FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = _FIGDIR / "split_time_breakdown.pdf"
    png = _FIGDIR / "split_time_breakdown.png"
    # Native canvas: 3.34 in wide.  600 dpi => 2004 px, satisfying the
    # >=600 dpi insertion requirement for this bitmap figure.
    info = save_native(build, pdf, SINGLE_COL_IN, png, png_dpi=600,
                       targets=SPLIT_TARGETS)
    report(info, "split_time_breakdown")

    # Rebuild to validate the plotted geometry, then check canvas consistency.
    fig, ax = _plot(rows, 1.0)
    validate(rows, ax)
    plt.close(fig)

    png_w, png_h = _png_size(png)
    page = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)", page)
    pdf_w, pdf_h = float(m.group(1)) / 72.0, float(m.group(2)) / 72.0

    assert abs(pdf_w - SINGLE_COL_IN) < 0.01, ("pdf width", pdf_w)
    assert abs(pdf_w - png_w / 600.0) < 0.01, ("pdf/png width mismatch", pdf_w, png_w)
    assert pdf_h <= 2.6 + 1e-6, ("canvas too tall", pdf_h)
    assert png_w >= 2004, ("png width < 2004 px", png_w)
    print(f"canvas OK: pdf {pdf_w:.3f} x {pdf_h:.3f} in; "
          f"png {png_w} x {png_h} px ({png_w / SINGLE_COL_IN:.0f} dpi @ 3.34 in)")

    for p in (pdf, png):
        print(f"wrote {p.relative_to(_ROOT)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
