#!/usr/bin/env python3
"""PyMuPDF bbox-gap / font / centring validator for the paper figures.

Each figure is read back from its *saved* PDF and every text item is
reconstructed with its bbox.  The script checks:

  * all horizontally adjacent text items on the same visual line have a
    non-negative gap (i.e. no two labels overlap);
  * x-axis tick labels *within a panel* have a gap >= MIN_TICK_GAP_PT
    (4 pt);
  * the effective (typeset) font size -- design size x insert/canvas -- stays
    at or above the 6.5 pt floor;
  * for ``frontier_all`` the three bottom panels are horizontally centred in
    the finished page: left and right whitespace differ by <= 5 % of the panel
    width, and the top/bottom rows have equal panel widths and aligned
    centres.

PyMuPDF merges tightly-kerned tick labels into a single span (robustness
renders as ``'1k2k4k8k'``), so spans are split back into per-tick tokens before
measuring.  Mathtext superscripts (``10`` + ``4``) are merged into a single
``10^4`` item so the exponent is checked as part of the label.

Adjacency is defined line-wise: two items are on the same line when their
vertical centres are within a fraction of the glyph height.

Exit status is non-zero if any check fails, so the script can gate a build::

    python scripts/verify_figure_gaps.py                 # all five figures
    python scripts/verify_figure_gaps.py --figure robustness
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figstyle import (  # noqa: E402
    DOUBLE_COL_IN,
    SINGLE_COL_IN,
    drop_scripts,
    pdf_font_sizes,
    pdf_width_in,
)

MIN_TEXT_GAP_PT = 0.0
MIN_TICK_GAP_PT = 4.0
MIN_FONT_PT = 6.5
CENTRE_TOL_FRAC = 0.05          # <= 5 % of the panel width

# name -> (pdf, insertion width)
FIGURES = {
    "frontier_all": ("frontier_all.pdf", DOUBLE_COL_IN),
    "lire_l_sweep_1M": ("lire_l_sweep_1M.pdf", SINGLE_COL_IN),
    "latency_stability_1M": ("latency_stability_1M.pdf", SINGLE_COL_IN),
    "tail_latency_CDF": ("tail_latency_CDF.pdf", DOUBLE_COL_IN),
    "robustness": ("robustness.pdf", SINGLE_COL_IN),
}

# decimal / integer / 10^k / k-suffixed tick labels
_TICK_RE = re.compile(r"^(?:[-+]?\d+(?:\.\d+)?|10\^-?\d+|\d+k)$")


def _split_tick_span(span: dict) -> list[dict]:
    """Split a merged run like ``'1k2k4k8k'`` into per-tick bboxes."""
    chars = span["chars"]
    text = "".join(c["c"] for c in chars)
    if not re.fullmatch(r"(?:\d+k\s*)+", text):
        return [{"text": text, "bbox": tuple(span["bbox"])}]

    items: list[dict] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch["c"].isspace():
            i += 1
            continue
        token = ch["c"]
        x0, y0, x1, y1 = ch["bbox"]
        if i + 1 < len(chars) and chars[i + 1]["c"] == "k":
            nxt = chars[i + 1]
            token += "k"
            x1 = max(x1, nxt["bbox"][2])
            y0 = min(y0, nxt["bbox"][1])
            y1 = max(y1, nxt["bbox"][3])
            i += 2
        else:
            i += 1
        items.append({"text": token, "bbox": (x0, y0, x1, y1)})
    return items


def _merge_superscripts(items: list[dict]) -> list[dict]:
    """Fold a ``10`` base and its superscript into one ``10^<x>`` item."""
    merged: list[dict] = []
    i = 0
    while i < len(items):
        cur = items[i]
        nxt = items[i + 1] if i + 1 < len(items) else None
        if (cur["text"] == "10" and nxt is not None
                and _is_superscript_of(cur, nxt)):
            x0 = min(cur["bbox"][0], nxt["bbox"][0])
            y0 = min(cur["bbox"][1], nxt["bbox"][1])
            x1 = max(cur["bbox"][2], nxt["bbox"][2])
            y1 = max(cur["bbox"][3], nxt["bbox"][3])
            merged.append({"text": cur["text"] + "^" + nxt["text"],
                           "bbox": (x0, y0, x1, y1),
                           "baseline": cur.get("baseline", y1),
                           "rotated": cur.get("rotated", False)})
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged


def _is_superscript_of(base: dict, sup: dict) -> bool:
    bx0, by0, bx1, by1 = base["bbox"]
    sx0, sy0, _, sy1 = sup["bbox"]
    if sx0 < bx1 - 0.5:                      # must start at the base's right edge
        return False
    if (sy1 - sy0) > 0.85 * (by1 - by0):     # must be smaller
        return False
    if sy1 > by0 + 0.9 * (by1 - by0):        # baseline above the base's mid
        return False
    return True


def _span_is_rotated(span: dict) -> bool:
    """True when the span's characters advance vertically (rotated label)."""
    chars = span["chars"]
    if len(chars) < 2:
        return False
    dx = chars[-1]["bbox"][0] - chars[0]["bbox"][0]
    dy = chars[-1]["bbox"][1] - chars[0]["bbox"][1]
    return abs(dy) > abs(dx)


def collect_items(page: pymupdf.Page) -> list[dict]:
    """All text items on the page, tick spans split and superscripts merged.

    Every item gets a ``baseline`` (the span origin y, used for line grouping)
    and a ``rotated`` flag (rotated labels are checked separately from
    horizontal same-line gaps, because a tall rotated y-axis label would
    otherwise chain-stack neighbouring y-tick labels into one "line").
    """
    items: list[dict] = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            line_items: list[dict] = []
            for span in line["spans"]:
                if not any(c["c"].strip() for c in span["chars"]):
                    continue
                rotated = _span_is_rotated(span)
                baseline = span.get("origin", (0.0, span["bbox"][3]))[1]
                for part in _split_tick_span(span):
                    part["baseline"] = baseline
                    part["rotated"] = rotated
                    line_items.append(part)
            items.extend(_merge_superscripts(line_items))
    return items


def find_panel_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Axes rectangles, deduplicated and sorted top-to-bottom, left-to-right."""
    page_rect = page.rect
    rects: list[pymupdf.Rect] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            r = item[1]
            w, h = r.x1 - r.x0, r.y1 - r.y0
            if w < 6 or h < 8:
                continue                     # tick marks / legend swatches
            if (w > 0.95 * page_rect.width and h > 0.95 * page_rect.height):
                continue                     # page background
            rects.append(r)
    unique: list[pymupdf.Rect] = []
    for r in rects:
        if all(abs(r.x0 - s.x0) > 1 or abs(r.y0 - s.y0) > 1 for s in unique):
            unique.append(r)
    unique.sort(key=lambda r: (round(r.y0, 0), r.x0))
    return unique


def group_rows(rects: list[pymupdf.Rect]) -> list[list[pymupdf.Rect]]:
    """Group panel rectangles into visual rows by their top edge."""
    rows: list[list[pymupdf.Rect]] = []
    for r in sorted(rects, key=lambda r: (round(r.y0, 0), r.x0)):
        for row in rows:
            if abs(row[0].y0 - r.y0) <= 2.0:
                row.append(r)
                break
        else:
            rows.append([r])
    for row in rows:
        row.sort(key=lambda r: r.x0)
    return rows


def same_line(a: dict, b: dict) -> bool:
    """True when two horizontal items share a text baseline.

    Grouping by the exact baseline (not the bbox centre) keeps stacked y-tick
    labels on separate lines; rotated labels are never horizontal neighbours.
    """
    if a.get("rotated") or b.get("rotated"):
        return False
    _, ay0, _, ay1 = a["bbox"]
    _, by0, _, by1 = b["bbox"]
    tol = 0.35 * min(ay1 - ay0, by1 - by0)
    return abs(a["baseline"] - b["baseline"]) <= tol


def _overlap_area(a: dict, b: dict) -> float:
    """Positive overlap area of two bboxes in pt^2 (0 when disjoint)."""
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def line_gaps(items: list[dict]) -> list[tuple]:
    """Adjacent (left, right, gap) triples for every visual text line."""
    out: list[tuple] = []
    used = [False] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i]["bbox"][0])
    for idx in order:
        if used[idx]:
            continue
        line = [idx]
        used[idx] = True
        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                if any(same_line(items[k], items[j]) for k in line):
                    line.append(j)
                    used[j] = True
                    changed = True
        line.sort(key=lambda i: items[i]["bbox"][0])
        for a, b in zip(line, line[1:]):
            gap = items[b]["bbox"][0] - items[a]["bbox"][2]
            out.append((items[a], items[b], gap))
    return out


def x_tick_items(items: list[dict], rect: pymupdf.Rect) -> list[dict]:
    """Tick labels directly below a panel's axes rectangle."""
    found = []
    for it in items:
        x0, y0, x1, _ = it["bbox"]
        cx = 0.5 * (x0 + x1)
        if not (rect.x0 - 3 <= cx <= rect.x1 + 3):
            continue
        if not (rect.y1 - 1 <= y0 <= rect.y1 + 16):
            continue
        if _TICK_RE.match(it["text"].strip()):
            found.append(it)
    found.sort(key=lambda it: it["bbox"][0])
    return found


def verify_figure(name: str, pdf: Path, insert_in: float,
                  verbose: bool = True) -> list[str]:
    """Run every check for one figure; return a list of failure messages."""
    failures: list[str] = []
    doc = pymupdf.open(pdf)
    page = doc[0]
    width_in = page.rect.width / 72.0
    height_in = page.rect.height / 72.0
    items = collect_items(page)
    rects = find_panel_rects(page)
    rows = group_rows(rects)

    if verbose:
        print(f"figure: {pdf.name}  size={width_in:.3f} x {height_in:.3f} in  "
              f"insert={insert_in:.2f} in  panels={len(rects)}  "
              f"text_items={len(items)}")
        print("=" * 74)

    # 1) every adjacent same-line text pair must not overlap
    worst = None
    for left, right, gap in line_gaps(items):
        if worst is None or gap < worst[2]:
            worst = (left, right, gap)
        if gap < MIN_TEXT_GAP_PT:
            failures.append(
                f"overlap: {left['text']!r}->{right['text']!r} gap={gap:.2f}pt")
    if verbose and worst is not None:
        print(f"[all text] min same-line gap = {worst[2]:.2f}pt "
              f"({worst[0]['text']!r} -> {worst[1]['text']!r})")
    if worst is None:
        failures.append("no text items found")

    # 1b) rotated labels (y-axis labels, row annotations) must not collide
    rotated = [it for it in items if it.get("rotated")]
    rot_hits = 0
    for i in range(len(rotated)):
        for j in range(i + 1, len(rotated)):
            area = _overlap_area(rotated[i], rotated[j])
            if area > 1.0:                 # > 1 pt^2 of real overlap
                rot_hits += 1
                failures.append(
                    f"rotated overlap: {rotated[i]['text']!r} vs "
                    f"{rotated[j]['text']!r} area={area:.1f}pt^2")
    if verbose:
        print(f"[rotated text] collisions = {rot_hits}")

    # 1c) horizontal alphabetic labels (titles, legend, axis names) must not
    #     overlap each other.  Numeric tick labels are excluded: their font
    #     bboxes legitimately touch when tick rows are stacked.
    alpha = [it for it in items
             if re.search(r"[A-Za-z]", it["text"])
             and not _TICK_RE.match(it["text"].strip())
             and not it.get("rotated")]
    alpha_hits = 0
    for i in range(len(alpha)):
        for j in range(i + 1, len(alpha)):
            area = _overlap_area(alpha[i], alpha[j])
            if area > 1.0:
                alpha_hits += 1
                failures.append(
                    f"text overlap: {alpha[i]['text']!r} vs "
                    f"{alpha[j]['text']!r} area={area:.1f}pt^2")
    if verbose:
        print(f"[alphabetic text] collisions = {alpha_hits}")

    # 2) x-tick labels per panel, gap >= 4 pt
    if verbose:
        print("-" * 74)
    for n, rect in enumerate(rects, 1):
        ticks = x_tick_items(items, rect)
        if len(ticks) < 2:
            failures.append(
                f"panel {n}: only {len(ticks)} x tick label(s) found")
            continue
        gaps = [ticks[i + 1]["bbox"][0] - ticks[i]["bbox"][2]
                for i in range(len(ticks) - 1)]
        if verbose:
            labels = ", ".join(t["text"] for t in ticks)
            min_gap = min(gaps)
            print(f"[panel {n}] x ticks: {labels}   min gap={min_gap:.2f}pt")
        for i, gap in enumerate(gaps):
            if gap < MIN_TICK_GAP_PT:
                failures.append(
                    f"panel {n}: x-tick {ticks[i]['text']!r}->"
                    f"{ticks[i + 1]['text']!r} gap={gap:.2f}pt "
                    f"< {MIN_TICK_GAP_PT}pt")

    # 3) effective font floor (design size x insert / canvas)
    factor = insert_in / pdf_width_in(pdf)
    sizes = drop_scripts(pdf_font_sizes(pdf))
    if not sizes:
        failures.append("no font sizes found")
        min_eff = float("nan")
    else:
        min_eff = min(sizes) * factor
        if min_eff < MIN_FONT_PT:
            failures.append(
                f"minimum effective font {min_eff:.2f}pt < {MIN_FONT_PT}pt")
    if verbose:
        print(f"[fonts] factor={factor:.4f}  "
              f"min effective={min_eff:.2f}pt")

    # 4) frontier_all: bottom-row centring / row alignment
    if name == "frontier_all" and len(rows) == 2:
        if verbose:
            print("-" * 74)
        top = max(rows, key=len)
        bottom = min(rows, key=len)
        left_margin = min(r.x0 for r in bottom)
        right_margin = page.rect.width - max(r.x1 for r in bottom)
        panel_w = sum(r.x1 - r.x0 for r in bottom) / len(bottom)
        diff = abs(left_margin - right_margin)
        tol = CENTRE_TOL_FRAC * panel_w
        top_c = 0.5 * (min(r.x0 for r in top) + max(r.x1 for r in top))
        bot_c = 0.5 * (min(r.x0 for r in bottom) + max(r.x1 for r in bottom))
        top_w = sum(r.x1 - r.x0 for r in top) / len(top)
        bot_w = panel_w
        if verbose:
            print(f"[centre] bottom left={left_margin:.2f}pt "
                  f"right={right_margin:.2f}pt diff={diff:.2f}pt "
                  f"(tol={tol:.2f}pt, panel_w={panel_w:.2f}pt)")
            print(f"[centre] top centre={top_c:.2f}pt bottom centre={bot_c:.2f}pt "
                  f"diff={abs(top_c - bot_c):.2f}pt; "
                  f"row widths top={top_w:.2f} bottom={bot_w:.2f}pt")
        if diff > tol:
            failures.append(
                f"frontier bottom row off-centre: left={left_margin:.2f}pt "
                f"right={right_margin:.2f}pt diff={diff:.2f}pt > {tol:.2f}pt")
        if abs(top_c - bot_c) > tol:
            failures.append(
                f"frontier row centres misaligned by {abs(top_c - bot_c):.2f}pt")
        if abs(top_w - bot_w) > tol:
            failures.append(
                f"frontier row widths differ by {abs(top_w - bot_w):.2f}pt")
    elif name == "frontier_all":
        failures.append(f"frontier_all: expected 2 rows, got {len(rows)}")

    if verbose:
        print("=" * 74)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, default=_ROOT / "figures")
    parser.add_argument("--figure", default="all",
                        choices=["all", *FIGURES.keys()])
    parser.add_argument("--pdf", type=Path, default=None,
                        help="verify a single arbitrary PDF (default 3.34 in)")
    parser.add_argument("--insert-in", type=float, default=None)
    args = parser.parse_args()

    all_failures: list[str] = []
    if args.pdf is not None:
        insert_in = args.insert_in if args.insert_in is not None else SINGLE_COL_IN
        all_failures += [f"{args.pdf.name}: {m}" for m in
                         verify_figure(args.pdf.stem, args.pdf, insert_in)]
    else:
        names = list(FIGURES) if args.figure == "all" else [args.figure]
        for name in names:
            pdf_name, insert_in = FIGURES[name]
            pdf = args.fig_dir / pdf_name
            all_failures += [f"{name}: {m}" for m in
                             verify_figure(name, pdf, insert_in)]

    print()
    if all_failures:
        print("FAILED:")
        for msg in all_failures:
            print(f"  - {msg}")
        return 1
    print(f"OK: all same-line gaps >= {MIN_TEXT_GAP_PT}pt; all x-tick gaps "
          f">= {MIN_TICK_GAP_PT}pt; all effective fonts >= {MIN_FONT_PT}pt; "
          f"frontier bottom row centred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
