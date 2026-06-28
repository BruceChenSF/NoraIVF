#!/usr/bin/env python3
"""Plot combined query/insert/delete P99 latency figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


QUERY_METHODS = ["NoraIVF", "SPFresh", "Frozen"]
MICRO_METHODS = {"NoraIVF": "Nora", "SPFresh": "LIRE", "Frozen": "Frozen"}
COLORS = {
    "NoraIVF": "#009E73",
    "SPFresh": "#0072B2",
    "Frozen": "#D55E00",
}
LINESTYLES = {
    "NoraIVF": "-",
    "SPFresh": "-",
    "Frozen": ":",
}
MARKERS = {
    "NoraIVF": "D",
    "SPFresh": "o",
    "Frozen": "^",
}


def micro_envelope(micro: dict, method: str, metric: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_method = MICRO_METHODS[method]
    traces = []
    sizes = None
    for dataset in micro["datasets"].values():
        values = dataset[source_method]
        if sizes is None:
            sizes = np.asarray(values["size"], dtype=float) / 1000.0
        traces.append(np.asarray(values[metric], dtype=float) * scale)
    arr = np.vstack(traces)
    return sizes, np.mean(arr, axis=0), np.min(arr, axis=0), np.max(arr, axis=0)


def plot(micro: dict, output_pdf: Path, output_png: Path | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.32, 2.95))
    panel_specs = [
        ("Query P99", "Latency (ms)", "query_p99_ms", 1.0, False),
        ("Insert P99", "Latency (ms)", "insert_p99_us", 0.001, True),
        ("Delete P99", "Latency (us)", "delete_p99_us", 1.0, True),
    ]

    for ax, (title, ylabel, metric, scale, log_y) in zip(axes, panel_specs):
        panel_values = []
        for method in QUERY_METHODS:
            x, mean, lo, hi = micro_envelope(micro, method, metric, scale=scale)
            if mean.size == 0:
                continue
            panel_values.extend([lo, hi])
            ax.fill_between(
                x,
                lo,
                hi,
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                label=method,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markersize=2.1,
                linewidth=1.05,
                markeredgewidth=0.0,
            )

        ax.set_title(title, pad=2.0)
        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.grid(True, which="major", linewidth=0.28, alpha=0.34)
        ax.tick_params(axis="both", which="major", pad=1.2, length=2.2)
        ax.set_xlim(4, 31)
        ax.set_xticks([5, 10, 20, 30])
        ax.set_xlabel("Index size (K vectors)", labelpad=2.0)
        if log_y:
            positive = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in panel_values])
            y_min = 10 ** np.floor(np.log10(np.min(positive)))
            y_max = 10 ** np.ceil(np.log10(np.max(positive)))
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=8))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80)
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, which="minor", axis="y", linewidth=0.18, alpha=0.20)
        else:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=0.40)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench", type=Path, default=Path("results/microbenchmark_four_methods_30k.json"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/exp5_latency_stability_combined.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/exp5_latency_stability_combined.png"))
    args = parser.parse_args()
    with args.microbench.open() as f:
        micro = json.load(f)
    plot(micro, args.pdf, args.png)
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
