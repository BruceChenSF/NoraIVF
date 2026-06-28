#!/usr/bin/env python3
"""Redraw recall-QPS frontiers with an explicit LIRE-L0 curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_ORDER = [
    "mnist-784-euclidean",
    "fashion-mnist-784-euclidean",
    "core50",
    "ScanNetCLIP",
    "Ego4D",
]

DATASET_LABELS = {
    "mnist-784-euclidean": "MNIST",
    "fashion-mnist-784-euclidean": "Fashion",
    "core50": "CoRE50",
    "ScanNetCLIP": "ScanNet",
    "Ego4D": "Ego4D",
}

METHOD_SPECS = [
    ("Frozen", "Frozen", "#d95f02", "^", ":"),
    ("Rebuild", "Rebuild", "#e6ab02", "P", "--"),
    ("DeDrift", "DeDrift", "#cc79a7", "v", "-."),
    ("SPFresh-L0", "LIRE-L0", "#666666", "x", (0, (2, 1))),
    ("SPFresh", "SPFresh", "#1f77b4", "o", "-"),
    ("NovaIVF", "NoraIVF", "#009e73", "D", "-"),
]


def load_results(paths: list[Path]) -> dict:
    merged = {}
    for path in paths:
        with path.open() as f:
            merged.update(json.load(f))
    return merged


def method_result(dataset_result: dict, method: str) -> dict:
    if method == "SPFresh-L0":
        return dataset_result["lire_l_sweep"][method]
    return dataset_result["methods"][method]


def collect_rows(results: dict) -> list[dict]:
    rows = []
    for dataset in DATASET_ORDER:
        dataset_result = results[dataset]
        for method, label, *_ in METHOD_SPECS:
            result = method_result(dataset_result, method)
            frontier = result["frontier"]
            for nprobe, recall, qps in zip(frontier["nprobe"], frontier["recall"], frontier["qps"]):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "label": label,
                        "nprobe": nprobe,
                        "recall": recall,
                        "qps": qps,
                    }
                )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(results: dict, output_pdf: Path, output_png: Path | None) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.25), sharey=False)
    legend_handles = []
    legend_labels = []
    for ax, dataset in zip(axes, DATASET_ORDER):
        dataset_result = results[dataset]
        for method, label, color, marker, linestyle in METHOD_SPECS:
            result = method_result(dataset_result, method)
            frontier = result["frontier"]
            line, = ax.plot(
                frontier["recall"],
                frontier["qps"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=3.5,
                label=label,
            )
            if dataset == DATASET_ORDER[0]:
                legend_handles.append(line)
                legend_labels.append(label)
        ax.set_title(DATASET_LABELS[dataset], pad=2)
        ax.set_xlabel("Recall")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.set_xlim(left=max(0.3, ax.get_xlim()[0]), right=1.01)
    axes[0].set_ylabel("QPS")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    results = load_results(args.inputs)
    rows = collect_rows(results)
    write_csv(rows, args.csv)
    plot(results, args.pdf, args.png)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print(f"Wrote {args.pdf}")
    if args.png:
        print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
