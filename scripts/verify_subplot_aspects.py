#!/usr/bin/env python3
"""PyMuPDF validator: every subplot's axes drawing area must be 1.2:1.

The paper unifies all multi-panel figures on a single subplot shape: the axes
drawing area (the rectangle bounded by the spines, i.e. the *plotting* region
excluding titles / tick labels / axis labels) is 1.2 wide for every 1.0 high,
within +/-0.1.  This is checked directly on the saved PDF: the axes patches are
recovered with :func:`verify_figure_gaps.find_panel_rects` and their width /
height ratio is measured in PDF points (the ratio is scale invariant, so the
design canvas vs. on-page insertion width does not matter).

Usage::

    python scripts/verify_subplot_aspects.py
    python scripts/verify_subplot_aspects.py --figure robustness

Exit status is non-zero if any panel deviates by more than ``TOL``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from verify_figure_gaps import find_panel_rects  # noqa: E402

_ROOT = _SCRIPT_DIR.parent

TARGET_AR = 1.2
TOL = 0.1

FIGURES = [
    ("lire_l_sweep_1M", "lire_l_sweep_1M.pdf"),
    ("frontier_all", "frontier_all.pdf"),
    ("latency_stability_1M", "latency_stability_1M.pdf"),
    ("tail_latency_CDF", "tail_latency_CDF.pdf"),
    ("robustness", "robustness.pdf"),
]


def measure(pdf: Path) -> list[float]:
    doc = pymupdf.open(pdf)
    page = doc[0]
    rects = find_panel_rects(page)
    ars = [((r.x1 - r.x0) / (r.y1 - r.y0)) for r in rects]
    return ars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=_ROOT / "figures")
    parser.add_argument("--figure", default="all",
                        choices=["all", *[n for n, _ in FIGURES]])
    args = parser.parse_args()

    names = [n for n, _ in FIGURES] if args.figure == "all" else [args.figure]
    failures: list[str] = []

    print(f"target axes aspect = {TARGET_AR:.2f} : 1  (tolerance +/-{TOL:.2f})")
    print(f"{'figure':24s} {'panels':>6s} {'min AR':>8s} {'max AR':>8s} "
          f"{'max|dev|':>9s}  status")
    print("-" * 74)

    for name in names:
        pdf = args.fig_dir / dict(FIGURES)[name]
        ars = measure(pdf)
        if not ars:
            failures.append(f"{name}: no panel rectangles found")
            print(f"{name:24s} {0:6d} {'-':>8s} {'-':>8s} {'-':>9s}  FAIL")
            continue
        devs = [abs(a - TARGET_AR) for a in ars]
        worst = max(devs)
        ok = worst <= TOL
        if not ok:
            failures.append(
                f"{name}: worst |AR-{TARGET_AR}|={worst:.3f} > {TOL} "
                f"(panels: {', '.join(f'{a:.3f}' for a in ars)})")
        print(f"{name:24s} {len(ars):6d} {min(ars):8.3f} {max(ars):8.3f} "
              f"{worst:9.3f}  {'OK' if ok else 'FAIL'}")

    print("-" * 74)
    if failures:
        print("FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(f"OK: every subplot axes aspect is {TARGET_AR:.2f}:1 "
          f"within +/-{TOL:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
