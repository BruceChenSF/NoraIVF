#!/usr/bin/env python3
"""Read back the generated figures and assert their effective type sizes.

For each figure the validator measures the *saved* artifact (not the in-memory
figure) and prints the design canvas width, the insertion width, and the
min/most-common font sizes found in the PDF content streams.  It then asserts:

  1. ``|canvas_in - insert_in| / insert_in <= 0.02``  (canvas == insertion)
  2. ``min_effective_font_pt >= 6.5``                 (hard floor)
  3. ``png_pixels / 3.34 >= 600``                     (bitmap resolution)

Mathtext script sizes (matplotlib renders them at 0.7x the parent) are dropped
before the font statistics are taken, so the reported minimum is a real text
size, not a superscript.

Run after the three plotting scripts::

    python scripts/plot_robustness.py
    python scripts/plot_tail_latency_cdf.py
    python scripts/plot_split_time_breakdown.py
    python scripts/verify_figure_sizes.py
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figstyle import (  # noqa: E402
    DOUBLE_COL_IN,
    SINGLE_COL_IN,
    drop_scripts,
    pdf_font_sizes,
    pdf_width_in,
)

_ROOT = Path(__file__).resolve().parent.parent

# (label, pdf name, insertion width, png name or None)
FIGURES = [
    ("robustness", "robustness.pdf", SINGLE_COL_IN, "robustness.png"),
    ("tail_latency_CDF", "tail_latency_CDF.pdf", DOUBLE_COL_IN,
     "tail_latency_CDF.png"),
    ("split_time_breakdown", "split_time_breakdown.pdf", SINGLE_COL_IN,
     "split_time_breakdown.png"),
]

MIN_FONT_PT = 6.5
MAX_WIDTH_ERROR = 0.02
MIN_PNG_DPI = 600.0
PNG_CHECK_WIDTH_IN = 3.34  # literal single-column width from the spec


def png_width_px(path: Path) -> int:
    """Pixel width from the PNG IHDR chunk (no image decode needed)."""
    with path.open("rb") as fh:
        sig = fh.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a PNG")
        length = struct.unpack(">I", fh.read(4))[0]
        chunk = fh.read(4)
        if chunk != b"IHDR":
            raise ValueError(f"{path}: first chunk is not IHDR")
        width = struct.unpack(">I", fh.read(4))[0]
        del length
    return width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=_ROOT / "figures")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"{'figure':22s} {'canvas(in)':>10s} {'insert(in)':>10s} "
          f"{'min(pt)':>8s} {'common(pt)':>18s} {'png(px)':>9s} {'dpi@3.34':>9s}")
    print("-" * 96)

    for label, pdf_name, insert_in, png_name in FIGURES:
        pdf = args.fig_dir / pdf_name
        canvas_in = pdf_width_in(pdf)
        sizes = drop_scripts(pdf_font_sizes(pdf))
        if not sizes:
            failures.append(f"{label}: no font sizes found in {pdf_name}")
            continue
        min_pt = min(sizes)
        common = ", ".join(f"{s:g}" for s, _ in Counter(sizes).most_common(3))

        png_px = None
        dpi_at_334 = None
        if png_name is not None:
            png = args.fig_dir / png_name
            png_px = png_width_px(png)
            dpi_at_334 = png_px / PNG_CHECK_WIDTH_IN

        print(f"{label:22s} {canvas_in:10.3f} {insert_in:10.3f} "
              f"{min_pt:8.2f} {common:>18s} "
              f"{(png_px if png_px is not None else 0):9d} "
              f"{(dpi_at_334 if dpi_at_334 is not None else float('nan')):9.1f}")

        err = abs(canvas_in - insert_in) / insert_in
        if err > MAX_WIDTH_ERROR:
            failures.append(
                f"{label}: |canvas-insert|/insert={err:.3%} > "
                f"{MAX_WIDTH_ERROR:.0%} (canvas {canvas_in:.3f} in vs "
                f"insert {insert_in:.3f} in)")
        if min_pt < MIN_FONT_PT:
            failures.append(
                f"{label}: minimum font {min_pt:.2f} pt < {MIN_FONT_PT} pt")
        if png_px is not None and dpi_at_334 < MIN_PNG_DPI:
            failures.append(
                f"{label}: PNG {png_px}px / 3.34in = {dpi_at_334:.1f} dpi < "
                f"{MIN_PNG_DPI:.0f} dpi")

    print("-" * 96)
    if failures:
        print("FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("OK: canvas width, font floor and PNG resolution all pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
