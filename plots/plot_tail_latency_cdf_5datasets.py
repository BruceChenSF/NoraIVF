#!/usr/bin/env python3
"""Plot compact five-dataset tail-latency CDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHODS = [
    ("NoraIVF", "NoraIVF", "#009E73", "-"),
    ("SPFresh", "SPFresh", "#0072B2", "-"),
    ("Frozen", "Frozen", "#D55E00", ":"),
]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["rows"]


def cdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=np.float64))
    ys = np.arange(1, len(xs) + 1, dtype=np.float64) / max(len(xs), 1)
    return xs, ys


def plot(rows: list[dict], output_pdf: Path, output_png: Path | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(3.32, 2.72))
    axes = axes.ravel()

    for ax, row in zip(axes[:5], rows):
        all_lats = []
        for method, label, color, style in METHODS:
            xs, ys = cdf(row[method]["query_latencies_ms"])
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.0, label=label)
            all_lats.extend(xs.tolist())
        upper = np.percentile(all_lats, 99.5) * 1.08 if all_lats else 1.0
        ax.set_xlim(left=0, right=max(upper, 0.5))
        ax.set_ylim(0, 1.02)
        ax.set_title(row["label"].replace("ScanNetCLIP", "ScanNet"), pad=1.5)
        ax.grid(True, linewidth=0.25, alpha=0.30)
        ax.tick_params(axis="both", which="major", pad=1.0, length=2.2)

    axes[0].set_ylabel("CDF")
    axes[2].set_ylabel("CDF")
    axes[4].set_ylabel("CDF")
    axes[4].set_xlabel("Latency (ms)")
    axes[3].set_xlabel("Latency (ms)")

    legend_ax = axes[5]
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=color, linestyle=style, linewidth=1.2, label=label)
        for _, label, color, style in METHODS
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.55,
    )

    fig.tight_layout(w_pad=0.55, h_pad=0.45)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/p99_stability_5datasets.json"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/exp7_tail_latency.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/exp7_tail_latency.png"))
    args = parser.parse_args()
    plot(load_rows(args.input), args.pdf, args.png)
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
