#!/usr/bin/env python3
"""Parameter robustness figure: cluster-count (nlist) sweep.

Compact single-column figure: ``figures/robustness.{pdf,png}`` (1x2 panels,
inserted at ``\\linewidth`` = 3.34 in).  Each panel is dual-axis:

  * left  axis -- cumulative maintenance time (log, s), solid method-coloured
                  curves with filled markers for the four stateful methods
                  (DeDrift / SPFresh / NoraIVF / AdaIVF).  Frozen maintenance
                  is exactly 0 s and cannot be drawn on a log axis, so it only
                  appears on the right axis.
  * right axis -- R@100 = recall@100 at the max-probe operating point
                  (nprobe=64), linear 0.80-1.00, dashed curves with open
                  markers for all five methods (incl. Frozen).

Panels cover SIFT-1M and AudioSet-2M, both over the full-coverage stream
(n_init = n_stream = half the corpus, ``gt_scope == "full"``), with four
nlist points {1024, 2048, 4096, 8192}; the common nlist=4096 operating point
is marked by a pale dashed vertical line.

Sources:
  * results/nlist_sweep_full/sift-128-euclidean_nlist{1024,2048,8192}.json
  * results/nlist_sweep_full/audioset-unbal-128_nlist{1024,2048,8192}.json
  * results/claim_evidence_{sift,audioset}.json  (nlist=4096 operating point,
    also the source of every Frozen maintenance/R@100 reading)

The expected numbers are asserted at import time: a silent data change stops
the build instead of producing a wrong figure.

Style (colors/markers) is reused from ``plot_claim_figures.py``; type sizes
are unified through ``paper_figstyle.py``.  The design canvas IS the insertion
width (3.34 in) and the PDF is not tight-cropped, so every point-based size is
the effective on-page size.  Every axes is locked to the paper subplot aspect
(1.2:1) via ``paper_figstyle.lock_aspect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
)
from matplotlib.ticker import NullLocator

# sibling plotter provides the shared method palette (single source)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_claim_figures import display_name, style_for  # noqa: E402
from paper_figstyle import (  # noqa: E402
    SINGLE_COL_IN,
    lock_aspect,
    report,
    save_native,
    scaled_rc,
)

_ROOT = Path(__file__).resolve().parent.parent
SWEEP = _ROOT / "results" / "nlist_sweep_full"

# Effective type sizes for the native single-column canvas (canvas == insert,
# so design sizes are already the on-page sizes).  Ticks/axis labels 7.0 pt,
# panel titles 8 pt, legend 7 pt, hard floor 6.5 pt.
ROBUST_TARGETS = {"tick": 7.0, "label": 7.0, "text": 6.5, "legend": 7.0}
MAIN = {
    "sift-128-euclidean": _ROOT / "results" / "claim_evidence_sift.json",
    "audioset-unbal-128": _ROOT / "results" / "claim_evidence_audioset.json",
}

MAIN_NLIST = 4096
NLIST_POINTS = (1024, 2048, 4096, 8192)
SWEEP_POINTS = (1024, 2048, 8192)          # 4096 comes from the main experiment
METHODS = ("Frozen", "DeDrift", "SPFresh", "NovaIVF", "AdaIVF")
MAINT_METHODS = ("DeDrift", "SPFresh", "NovaIVF", "AdaIVF")  # Frozen is 0 s
NPROBE = 64

# --------------------------------------------------------------------------- #
# expected values (maintenance_time_s and R@100 in NLIST_POINTS order) -- the
# build aborts if any point disagrees.  Maintenance tolerance 0.05 s (values
# are quoted to 0.1 s), recall tolerance 6e-4 (values are quoted to 3 dp).
# --------------------------------------------------------------------------- #
EXPECTED = {
    "sift-128-euclidean": {
        "SPFresh": {"maint": (338.8, 697.7, 1496.6, 3809.8),
                    "r64": (0.982, 0.954, 0.911, 0.850)},
        "NovaIVF": {"maint": (13.8, 14.4, 14.8, 19.5),
                    "r64": (0.980, 0.950, 0.904, 0.844)},
        "AdaIVF":  {"maint": (41.0, 61.7, 53.3, 85.4),
                    "r64": (0.985, 0.963, 0.919, 0.853)},
        "DeDrift": {"maint": (162.5, 100.8, 51.5, 48.1),
                    "r64": (0.964, 0.930, 0.876, 0.814)},
    },
    "audioset-unbal-128": {
        "SPFresh": {"maint": (706.6, 1887.6, 4589.3, 7271.8),
                    "r64": (0.994, 0.990, 0.982, 0.968)},
        "NovaIVF": {"maint": (38.8, 41.9, 74.3, 43.7),
                    "r64": (0.992, 0.989, 0.981, 0.966)},
        "AdaIVF":  {"maint": (83.3, 75.6, 81.8, 90.6),
                    "r64": (0.994, 0.991, 0.983, 0.970)},
        "DeDrift": {"maint": (537.7, 319.9, 267.1, 194.4),
                    "r64": (0.983, 0.973, 0.961, 0.945)},
    },
}
MAINT_TOL = 0.05
R64_TOL = 6e-4


def _r64(entry: dict, method: str) -> float:
    """Recall@100 at nprobe=64 (the frontier's max-probe point)."""
    frontier = entry["methods"][method]["frontier"]
    nprobes = frontier["nprobe"]
    assert NPROBE in nprobes, (method, nprobes)
    idx = nprobes.index(NPROBE)
    return float(frontier["recall"][idx])


def _row(entry: dict, method: str) -> dict[str, float]:
    return {"maint": float(entry["methods"][method]["maintenance_time_s"]),
            "r64": _r64(entry, method)}


def load_nlist_points(ds_key: str) -> dict[int, dict[str, dict]]:
    """nlist -> {method: {'maint': s, 'r64': recall}} (sweep + main 4096)."""
    points: dict[int, dict[str, dict]] = {}
    for nl in SWEEP_POINTS:
        path = SWEEP / f"{ds_key}_nlist{nl}.json"
        with path.open(encoding="utf-8") as fh:
            entry = json.load(fh)[ds_key]
        assert int(entry["nlist"]) == nl, (path, entry["nlist"])
        # full-coverage stream: n_init ~= n_stream (each half the corpus)
        assert entry.get("gt_scope") == "full", (path, entry.get("gt_scope"))
        assert abs(entry["n_init"] - entry["n_stream"]) <= 0.01 * entry["n_init"], (
            path, entry["n_init"], entry["n_stream"])
        points[nl] = {m: _row(entry, m) for m in METHODS}

    with MAIN[ds_key].open(encoding="utf-8") as fh:
        entry = json.load(fh)[ds_key]
    assert int(entry["nlist"]) == MAIN_NLIST, entry["nlist"]
    points[MAIN_NLIST] = {m: _row(entry, m) for m in METHODS}

    for nl in NLIST_POINTS:
        missing = [m for m in METHODS if m not in points[nl]]
        if missing:
            raise KeyError(f"{ds_key} nlist={nl}: missing {missing}")
    return points


def assert_expected(ds_key: str, points: dict[int, dict[str, dict]]) -> None:
    """Fail loudly if any plotted point differs from the paper values."""
    for method, exp in EXPECTED[ds_key].items():
        for i, nl in enumerate(NLIST_POINTS):
            got = points[nl][method]
            assert abs(got["maint"] - exp["maint"][i]) <= MAINT_TOL, (
                ds_key, method, nl, "maint", got["maint"], exp["maint"][i])
            assert abs(got["r64"] - exp["r64"][i]) <= R64_TOL, (
                ds_key, method, nl, "r64", got["r64"], exp["r64"][i])


def short_nlist(nl: int) -> str:
    """Compact nlist tick label (1024 -> '1k', 8192 -> '8k')."""
    if nl % 1024 == 0:
        return f"{nl // 1024}k"
    return f"{nl / 1000:g}k"


def plot_nlist_panel(ax, points: dict, title: str, scale: float):
    """Left axis = maintenance time (solid); right twin = R@100 (dashed)."""
    nlists = sorted(points)

    # left axis: cumulative maintenance time (log, solid, filled markers)
    for method in MAINT_METHODS:
        color, marker, _ls = style_for(method)
        ax.plot(nlists, [points[nl][method]["maint"] for nl in nlists],
                color=color, marker=marker, linestyle="-",
                markerfacecolor=color, markeredgecolor=color,
                linewidth=1.1 * scale, markersize=2.8 * scale)
    # nlist=4096 main-experiment operating point (pale dashed guide)
    ax.axvline(MAIN_NLIST, color="#c8c8c8", linestyle=(0, (3, 2)),
               linewidth=0.9 * scale, zorder=0)

    ax.set_title(title, pad=3.0 * scale)
    ax.set_xlabel("nlist", labelpad=1.0 * scale)
    # nlist is a power-of-two knob: log2 spacing puts the four values evenly
    # apart while the labels keep the 1k/2k/4k/8k reading.
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(nlists)
    ax.set_xticklabels([short_nlist(nl) for nl in nlists])
    ax.set_xlim(nlists[0] / 1.35, nlists[-1] * 1.35)
    ax.set_yscale("log")
    ax.set_ylim(10.0, 1e4)
    ax.set_yticks([1e1, 1e2, 1e3, 1e4])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", linewidth=0.28 * scale, alpha=0.3)
    ax.tick_params(axis="both", which="major", pad=1.5 * scale,
                   length=2.5 * scale)

    # right axis: R@100 (linear 0.80-1.00, dashed, open markers)
    axr = ax.twinx()
    for method in METHODS:
        color, marker, _ls = style_for(method)
        axr.plot(nlists, [points[nl][method]["r64"] for nl in nlists],
                 color=color, marker=marker, linestyle="--",
                 markerfacecolor="white", markeredgecolor=color,
                 linewidth=1.0 * scale, markersize=2.8 * scale)
    axr.set_ylim(0.80, 1.00)
    axr.set_yticks([0.8, 0.9, 1.0])
    axr.tick_params(axis="y", pad=1.5 * scale, length=2.5 * scale)
    return axr


# --------------------------------------------------------------------------- #
# two-row horizontal legend (row 1 = maintenance, row 2 = R@100).
#
# ``fig.legend`` cannot put a group label to the *left* of its entries, so the
# rows are packed by hand with offset boxes.  Every point-based size below is
# in native-canvas points (the canvas IS the insertion width, so these are the
# effective on-page sizes).  The 5-item R@100 row only fits the 3.34 in canvas
# with a compact marker-dominated handle (``HANDLE_W``), which the task
# explicitly allows in preference to dropping below the 7 pt legend size.
# --------------------------------------------------------------------------- #
LEGEND_FS = 7.0          # legend type size (hard floor 6.5 pt, unused)
HANDLE_W = 6.0           # handle box width (pt): marker + a short line
HANDLE_TEXT_PAD = 0.8    # handle -> label pad (pt)
ROW_GAP = 2.5            # label/entry separation along a row (pt, >= 2 pt)

# --------------------------------------------------------------------------- #
# Vertical layout of the compact canvas.
#
# The legend rows are anchored by absolute point offsets from the *top* edge,
# not by canvas fractions: when the canvas height shrinks (to delete the dead
# band that a 5-row legend left behind) the two rows must keep their original
# 3.42 pt separation instead of being squashed together.  With these numbers
# the canvas is exactly the stack of
#
#   top margin            1.764 pt
#   legend block         19.516 pt  (row 8.05 + 3.42 gap + row 8.05)
#   legend/panel gap      7.000 pt
#   title above axes      9.078 pt
#   axes drawing height  62.508 pt  (AR 1.2)
#   x tick/label band    27.399 pt
#   -------------------------------
#   total               127.265 pt = 1.7676 in
#
# ``LEGEND_BAND_TOP`` is the *fraction* of the canvas reserved by
# ``tight_layout`` for the title band; it is tuned (at the 3.34 in width and
# the height above) so the panels keep the original 27.399 pt bottom / 62.508
# pt height and the legend bottom lands 7 pt above the panel titles.
# --------------------------------------------------------------------------- #
FIG_H_IN = 1.7676              # canvas height (in); was 2.45 with 5-row legend
LEGEND_TOP_MARGIN_PT = 1.764   # canvas top -> top edge of legend row 1
LEGEND_ROW_SEP_PT = 11.466     # top of row 1 -> top of row 2 (row + 3.42 gap)
LEGEND_BAND_TOP = 0.8732       # tight_layout rect top (title band)


def _legend_handle(method: str, linestyle: str, filled: bool):
    """One compact legend swatch (marker over a short line segment)."""
    color, marker, _ls = style_for(method)
    height = LEGEND_FS * 1.15
    box = DrawingArea(HANDLE_W, height, 0, 0)
    y = height / 2.0
    box.add_artist(Line2D([0.6, HANDLE_W - 0.6], [y, y], color=color,
                          linestyle=linestyle, linewidth=1.05))
    box.add_artist(Line2D([HANDLE_W / 2.0], [y], color=color, marker=marker,
                          linestyle="None",
                          markerfacecolor=color if filled else "white",
                          markeredgecolor=color, markersize=3.0))
    return box


def _legend_item(method: str, linestyle: str, filled: bool):
    """A (handle, name) pair that participates in the horizontal row."""
    return HPacker(
        children=[_legend_handle(method, linestyle, filled),
                  TextArea(display_name(method),
                           textprops=dict(fontsize=LEGEND_FS, family="serif"))],
        align="center", pad=0, sep=HANDLE_TEXT_PAD)


def _legend_row(methods, linestyle: str, filled: bool, label: str):
    """Row = group label followed by horizontally packed method entries."""
    children = [TextArea(label, textprops=dict(fontsize=LEGEND_FS,
                                               family="serif"))]
    children += [_legend_item(m, linestyle, filled) for m in methods]
    return HPacker(children=children, align="center", pad=0, sep=ROW_GAP)


def build_robustness(scale: float = 1.0) -> plt.Figure:
    """Build the single-column 1x2 robustness figure."""
    del scale  # native canvas: fonts are already at the effective sizes
    s = 1.0
    plt.rcParams.update(scaled_rc(s, ROBUST_TARGETS))

    # Native single-column canvas: the page is NOT tight-cropped, so its width
    # equals the 3.34 in insertion width and the fonts below are effective.
    # The height is the legend + panels stack computed above (no dead band).
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL_IN, FIG_H_IN))

    datasets = [
        ("sift-128-euclidean", "SIFT-1M"),
        ("audioset-unbal-128", "AudioSet-2M"),
    ]
    right_axes = []
    for i, (ds_key, title) in enumerate(datasets):
        points = load_nlist_points(ds_key)
        assert_expected(ds_key, points)
        right_axes.append(plot_nlist_panel(axes[i], points, title, s))

    axes[0].set_ylabel("Maintenance time (s)", labelpad=1.0 * s)
    right_axes[1].set_ylabel("R@100", labelpad=1.0 * s)

    # two-row legend: row 1 = maintenance (solid/filled), row 2 = R@100
    # (dashed/open); entries run horizontally, group labels left-aligned.
    legend_boxes = []
    for label, methods, linestyle, filled in (
            ("Maint. (left)", MAINT_METHODS, "-", True),
            ("R@100 (right)", METHODS, "--", False)):
        box = AnchoredOffsetbox(
            loc="upper left", child=_legend_row(methods, linestyle, filled, label),
            frameon=False, pad=0, borderpad=0,
            bbox_to_anchor=(0.0, 1.0), bbox_transform=fig.transFigure)
        fig.add_artist(box)
        legend_boxes.append(box)

    # Centre the block on the 3.34 in canvas while keeping both group labels
    # left-aligned: measure the packed widths, then anchor both rows at the
    # same left edge.  A throwaway draw gives the TextArea metrics.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [b.get_window_extent(renderer).width for b in legend_boxes]
    left_frac = 0.5 * (fig.bbox.width - max(widths)) / fig.bbox.width
    h_pt = fig.get_figheight() * 72.0
    row_anchors = ((h_pt - LEGEND_TOP_MARGIN_PT) / h_pt,
                   (h_pt - LEGEND_TOP_MARGIN_PT - LEGEND_ROW_SEP_PT) / h_pt)
    for box, y in zip(legend_boxes, row_anchors):
        box.set_bbox_to_anchor((left_frac, y), transform=fig.transFigure)
    fig.canvas.draw()

    lock_aspect(axes)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, LEGEND_BAND_TOP), w_pad=0.9,
                     pad=0.3)

    # Geometry guard: the whole point of the compressed canvas is that the
    # legend sits just above the panels (single-digit pt gap) while the panels
    # keep the paper subplot aspect.  Fail the build if either drifts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pt = ((legend_boxes[1].get_window_extent(renderer).y0
               - axes[0].title.get_window_extent(renderer).y1)
              * 72.0 / fig.dpi)
    ax_bb = axes[0].get_window_extent(renderer)
    assert 5.5 <= gap_pt <= 8.5, f"legend/panel gap {gap_pt:.2f} pt out of band"
    assert abs(ax_bb.width / ax_bb.height - 1.2) < 0.01, "panel aspect drift"
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "figures")
    args = parser.parse_args()

    info = save_native(
        build_robustness, args.out_dir / "robustness.pdf", SINGLE_COL_IN,
        args.out_dir / "robustness.png", png_dpi=600, targets=ROBUST_TARGETS)
    report(info, "robustness")

    for ds_key, title in (("sift-128-euclidean", "SIFT-1M"),
                          ("audioset-unbal-128", "AudioSet-2M")):
        pts = load_nlist_points(ds_key)
        assert_expected(ds_key, pts)
        print(f"  {title}: nlist={sorted(pts)}  "
              f"maint_curves={len(MAINT_METHODS)}  r64_curves={len(METHODS)}")
    print(f"Wrote robustness.pdf/.png -> {args.out_dir}")


if __name__ == "__main__":
    main()
