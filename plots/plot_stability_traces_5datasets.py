#!/usr/bin/env python3
"""Plot compact five-dataset workload stability envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


METHODS = ["NoraIVF", "SPFresh", "Frozen"]
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


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["rows"]


def windowed(values: list[float], n_windows: int, reducer: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])

    n = min(n_windows, arr.size)
    chunks = [chunk for chunk in np.array_split(arr, n) if chunk.size]
    y = np.asarray([reducer(chunk) for chunk in chunks], dtype=float)
    x = np.linspace(0, 100, len(y))
    return x, y


def padded_ylim(series: list[np.ndarray], floor_zero: bool = False) -> tuple[float, float]:
    vals = np.concatenate([s[np.isfinite(s)] for s in series if s.size])
    if vals.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if floor_zero:
        lo = 0.0
    if np.isclose(lo, hi):
        pad = max(abs(hi) * 0.05, 0.01)
    else:
        pad = (hi - lo) * 0.12
    return (lo - pad if not floor_zero else 0.0, hi + pad)


def collect_metric(
    rows: list[dict],
    method: str,
    field: str,
    reducer: Callable[[np.ndarray], float],
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traces = []
    x_ref: np.ndarray | None = None
    for row in rows:
        x, y = windowed(row[method][field], n_windows, reducer)
        if y.size == 0:
            continue
        if x_ref is None:
            x_ref = x
        traces.append(y)
    if x_ref is None or not traces:
        return np.array([]), np.array([]), np.array([]), np.array([])

    min_len = min(len(t) for t in traces)
    arr = np.vstack([t[:min_len] for t in traces])
    x = x_ref[:min_len]
    return x, np.mean(arr, axis=0), np.min(arr, axis=0), np.max(arr, axis=0)


def plot(rows: list[dict], output_pdf: Path, output_png: Path | None) -> None:
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

    metrics = [
        ("recall_trace", "Recall@100", lambda a: float(np.mean(a)), False, "%.2f"),
        ("batch_qps", "QPS", lambda a: float(np.mean(a)), False, "%.0f"),
        ("query_latencies_ms", "P99 (ms)", lambda a: float(np.percentile(a, 99)), True, "%.1f"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(3.32, 2.85), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (field, ylabel, reducer, floor_zero, yfmt) in zip(axes, metrics):
        metric_series: list[np.ndarray] = []
        for method in METHODS:
            x, mean, lo, hi = collect_metric(rows, method, field, reducer, 12)
            if mean.size == 0:
                continue
            metric_series.extend([lo, hi])
            ax.fill_between(
                x,
                lo,
                hi,
                color=COLORS[method],
                alpha=0.10,
                linewidth=0.0,
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

        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.set_ylim(*padded_ylim(metric_series, floor_zero=floor_zero))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(yfmt))
        ax.grid(True, linewidth=0.28, alpha=0.34)
        ax.tick_params(axis="both", which="major", pad=1.2, length=2.2)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 50, 100])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    axes[-1].set_xlabel("Workload progress after warmup (%)", labelpad=2.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=0.42)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/p99_stability_5datasets.json"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/exp5_stability_traces_5datasets.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/exp5_stability_traces_5datasets.png"))
    args = parser.parse_args()
    plot(load_rows(args.input), args.pdf, args.png)
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
