#!/usr/bin/env python3
"""Plot a compact five-dataset stability summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["NoraIVF", "SPFresh", "Frozen"]
COLORS = {
    "NoraIVF": "#009E73",
    "SPFresh": "#0072B2",
    "Frozen": "#D55E00",
}


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["rows"]


def plot(rows: list[dict], output_pdf: Path, output_png: Path | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [row["label"].replace("ScanNetCLIP", "ScanNet") for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    offsets = {
        "NoraIVF": -width,
        "SPFresh": 0.0,
        "Frozen": width,
    }
    metrics = [
        ("avg_recall", "Recall@100", "Recall@100"),
        ("avg_qps", "Throughput", "QPS"),
        ("p99_ms", "Tail latency", "P99 (ms)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(3.32, 3.15), sharex=True)
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for method in METHODS:
            values = [row[method][metric] for row in rows]
            ax.bar(
                x + offsets[method],
                values,
                width,
                label=method,
                color=COLORS[method],
                linewidth=0.25,
                edgecolor="black",
            )
        ax.set_title(title, pad=2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(axis="y", linewidth=0.28, alpha=0.32)
        ax.tick_params(axis="both", which="major", pad=1.5, length=2.5)
        ax.margins(x=0.02)

    axes[0].set_ylim(0.44, 0.70)
    axes[2].set_ylim(0, max(row["Frozen"]["p99_ms"] for row in rows) * 1.15)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.6,
        handlelength=0.9,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.55)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/p99_stability_5datasets.json"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/exp5_stability_summary_5datasets.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/exp5_stability_summary_5datasets.png"))
    args = parser.parse_args()
    plot(load_rows(args.input), args.pdf, args.png)
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
