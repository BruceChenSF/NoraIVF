#!/usr/bin/env python3
"""Plot LIRE L-sweep trends from claim-evidence JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_l(method_name: str) -> int:
    match = re.search(r"L(\d+)$", method_name)
    if not match:
        raise ValueError(f"Cannot parse L from method name: {method_name}")
    return int(match.group(1))


def recall_at_nprobe(result: dict, nprobe: int) -> float:
    frontier = result["frontier"]
    nprobes = frontier["nprobe"]
    if nprobe in nprobes:
        idx = nprobes.index(nprobe)
    else:
        idx = min(range(len(nprobes)), key=lambda i: abs(nprobes[i] - nprobe))
    return float(frontier["recall"][idx])


def collect_rows(result_json: Path, nprobe: int) -> list[dict]:
    with result_json.open() as f:
        all_results = json.load(f)

    rows = []
    for dataset, dataset_result in all_results.items():
        for method, result in dataset_result["lire_l_sweep"].items():
            npa_rate = result.get("npa", {}).get("violation_rate")
            rows.append(
                {
                    "dataset": dataset,
                    "L": parse_l(method),
                    "method": method,
                    "family": "LIRE",
                    "reassignments": int(result["n_reassigns"]),
                    "maintenance_time_s": float(result["maintenance_time_s"]),
                    "npa_violation_rate": npa_rate,  # None when NPA was skipped
                    "recall_at_nprobe": recall_at_nprobe(result, nprobe),
                }
            )
        for method, result in dataset_result.get("lire_l_sweep_baselines", {}).items():
            npa_rate = result.get("npa", {}).get("violation_rate")
            rows.append(
                {
                    "dataset": dataset,
                    "L": None,
                    "method": method,
                    "family": "baseline",
                    "reassignments": int(result["n_reassigns"]),
                    "maintenance_time_s": float(result["maintenance_time_s"]),
                    "npa_violation_rate": npa_rate,
                    "recall_at_nprobe": recall_at_nprobe(result, nprobe),
                }
            )
    return sorted(rows, key=lambda row: (row["dataset"], row["family"], row["L"] if row["L"] is not None else -1))


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict], output_pdf: Path, output_png: Path | None) -> None:
    dataset_labels = {
        "mnist-784-euclidean": "MNIST",
        "fashion-mnist-784-euclidean": "Fashion",
        "core50": "CoRE50",
        "ScanNetCLIP": "ScanNet",
        "Ego4D": "Ego4D",
        "sift-128-euclidean": "SIFT-1M",
        "gist-960-euclidean": "GIST-1M",
    }
    dataset_order = ["mnist-784-euclidean", "fashion-mnist-784-euclidean",
                     "core50", "ScanNetCLIP", "Ego4D",
                     "sift-128-euclidean", "gist-960-euclidean"]
    present = {row["dataset"] for row in rows}
    datasets = [dataset for dataset in dataset_order if dataset in present]
    datasets.extend(sorted(present - set(datasets)))
    by_dataset = {
        dataset: [row for row in rows if row["dataset"] == dataset and row["family"] == "LIRE"]
        for dataset in datasets
    }
    baselines = {
        dataset: [row for row in rows if row["dataset"] == dataset and row["family"] == "baseline"]
        for dataset in datasets
    }

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

    metrics_all = [
        ("reassignments", "Reassignments", r"Vectors ($10^3$)"),
        ("maintenance_time_s", "Maintenance", "Time (s)"),
        ("npa_violation_rate", "NPA violation", "Rate (%)"),
        ("recall_at_nprobe", "Quality", "Recall@100 (%)"),
    ]
    # drop metrics with no data (e.g. NPA skipped via --skip-npa -> all None)
    metrics = [
        (key, title, ylabel)
        for key, title, ylabel in metrics_all
        if any(row.get(key) is not None for row in rows)
    ]
    n_metrics = len(metrics)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.32, 1.5 * n_rows + 0.05), sharex=True)
    axes = [axes] if n_metrics == 1 else axes.ravel()

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = {}
    l_values = sorted({row["L"] for row in rows if row["L"] is not None})

    def display_value(key: str, value: float | None) -> float | None:
        if value is None:
            return None
        if key == "reassignments":
            return value / 1000.0
        if key in {"npa_violation_rate", "recall_at_nprobe"}:
            return value * 100.0
        return value

    for ax, (key, title, ylabel) in zip(axes, metrics):
        for idx, dataset in enumerate(datasets):
            ds_rows = by_dataset[dataset]
            xs = [row["L"] for row in ds_rows]
            ys = [display_value(key, row[key]) for row in ds_rows]
            # keep only pairs where the metric is present
            xs = [x for x, y in zip(xs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not ys:
                continue
            line, = ax.plot(xs, ys, marker=markers[idx % len(markers)], linewidth=1.1,
                            markersize=2.8, label=dataset_labels.get(dataset, dataset))
            colors.setdefault(dataset, line.get_color())
            for baseline in baselines.get(dataset, []):
                baseline_y = display_value(key, baseline[key])
                if baseline_y is None:
                    continue
                ax.plot(l_values, [baseline_y] * len(l_values),
                        color=colors[dataset], linestyle="--", linewidth=0.9,
                        alpha=0.85, label="_nolegend_")
        ax.set_title(title, pad=1.5)
        ax.set_xlabel("L")
        ax.set_ylabel(ylabel)
        if key == "maintenance_time_s":
            ax.set_yscale("log")
            ax.set_ylabel("Time (s)")
        ax.grid(True, linewidth=0.28, alpha=0.30)
        ax.set_xticks(l_values)
        ax.tick_params(axis="both", which="major", pad=1.5, length=2.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.995), columnspacing=0.45, handlelength=0.9,
               handletextpad=0.25, labelspacing=0.0, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, 0.915), w_pad=0.72, h_pad=0.65)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--nprobe", type=int, default=8)
    args = parser.parse_args()

    rows = collect_rows(args.input, args.nprobe)
    if not rows:
        raise SystemExit("No L-sweep rows found.")
    write_csv(rows, args.csv)
    plot_rows(rows, args.pdf, args.png)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print(f"Wrote {args.pdf}")
    if args.png:
        print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
