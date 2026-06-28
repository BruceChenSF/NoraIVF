#!/usr/bin/env python3
"""
Explore the ability boundary of CDKM under the same local scope as LIRE.

After a Nora split, Local-CDKM-L optimizes the affected local subproblem over
the old cluster, the new cluster, and the top-L neighboring clusters.  Unlike
the boundary-sampling diagnostic, this gives CDKM the same local visibility as
LIRE-L and allows ownership moves inside that local scope.  The experiment
compares whether multi-cluster coordinate descent can match or beat LIRE with
the same L.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from experiments.run_claim_evidence_annbench import (
    L2NovaIVF,
    NPY_DATASETS,
    build_after_stream,
    squared_l2_to_centroids,
    summarize_index,
)


class L2LocalCDKMRepairNovaIVF(L2NovaIVF):
    def __init__(self, *args, local_l=5, local_iter=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_l = int(local_l)
        self.local_iter = int(local_iter)
        self.total_local_cdkm_time = 0.0
        self.n_local_candidates = 0

    def _split(self, j):
        before_splits = self.n_splits
        super()._split(j)
        if self.n_splits == before_splits:
            return
        _, new_c, _, _ = self.split_history[-1]
        t0 = time.perf_counter()
        self._local_cdkm_repair(j, new_c)
        dt = time.perf_counter() - t0
        self.total_local_cdkm_time += dt
        self.total_split_time += dt

    def _affected_clusters(self, c1, c2):
        center = 0.5 * (self.centroids[c1] + self.centroids[c2])
        active = [c for c in range(self.k) if c not in (c1, c2) and len(self.posting_lists[c]) > 0]
        if not active or self.local_l <= 0:
            return [c1, c2]
        dists = [(float(squared_l2_to_centroids(center, self.centroids[c:c + 1])[0]), c) for c in active]
        dists.sort(key=lambda item: item[0])
        return [c1, c2] + [c for _, c in dists[: min(self.local_l, len(dists))]]

    def _move_delta(self, idx, from_c, to_c):
        if from_c == to_c or self.n_j[from_c] <= 1:
            return 0.0
        vec = self.id_to_vec[idx]
        rho = float(np.dot(vec, vec))

        n_from = float(self.n_j[from_c])
        s_from = float(self.s_j[from_c])
        a_from = self.a_j[from_c]
        s_from_new = s_from - 2.0 * float(np.dot(vec, a_from)) + rho

        n_to = float(self.n_j[to_c])
        s_to = float(self.s_j[to_c])
        a_to = self.a_j[to_c]
        s_to_new = s_to + 2.0 * float(np.dot(vec, a_to)) + rho

        before = s_from / n_from + (s_to / n_to if n_to > 0 else 0.0)
        after = s_from_new / (n_from - 1.0) + s_to_new / (n_to + 1.0)
        return after - before

    def _local_cdkm_repair(self, c1, c2):
        affected = self._affected_clusters(c1, c2)
        affected_set = set(affected)
        candidates = []
        for c in affected:
            candidates.extend(list(self.posting_lists[c]))
        self.n_local_candidates += len(candidates)

        for _ in range(self.local_iter):
            changed = False
            for idx in list(candidates):
                if idx not in self.id_to_vec or idx not in self.ppm:
                    continue
                from_c = self.ppm[idx]
                if from_c not in affected_set:
                    continue

                best_c = from_c
                best_delta = 1e-9
                for to_c in affected:
                    if to_c == from_c:
                        continue
                    delta = self._move_delta(idx, from_c, to_c)
                    if delta > best_delta:
                        best_delta = delta
                        best_c = to_c

                if best_c != from_c:
                    self._move_vector(idx, from_c, best_c, self.id_to_vec[idx])
                    self.n_reassigns += 1
                    changed = True
            if not changed:
                break


def load_dataset(dataset_result, data_dir, npy_paths):
    name = dataset_result["dataset"]
    max_train = int(dataset_result["n_train"])
    max_queries = int(dataset_result["n_queries"])
    k = int(dataset_result["k"])
    if dataset_result["format"] == "hdf5":
        path = Path(data_dir) / f"{name}.hdf5"
        return load_ann_benchmark_hdf5(path, max_train=max_train, max_queries=max_queries, k=k)
    if dataset_result["format"] == "npy":
        return load_npy_vector_dataset_l2(npy_paths[name], max_train=max_train, max_queries=max_queries, k=k)
    raise ValueError(f"Unsupported dataset format: {dataset_result['format']}")


def recall_at_nprobe(result, nprobe):
    frontier = result["frontier"]
    nprobes = frontier["nprobe"]
    idx = nprobes.index(nprobe) if nprobe in nprobes else min(
        range(len(nprobes)), key=lambda i: abs(nprobes[i] - nprobe)
    )
    return float(frontier["recall"][idx])


def add_row(rows, dataset, method, family, l_value, result, nprobe):
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "family": family,
            "L": int(l_value) if l_value is not None else None,
            "reassignments": int(result["n_reassigns"]),
            "maintenance_time_s": float(result["maintenance_time_s"]),
            "npa_violation_rate": float(result["npa"]["violation_rate"]),
            "recall_at_nprobe": recall_at_nprobe(result, nprobe),
            "local_candidates": int(result.get("n_local_candidates", 0)),
            "local_cdkm_time_s": float(result.get("local_cdkm_time_s", 0.0)),
        }
    )


def run_diagnostic(args):
    npy_paths = dict(NPY_DATASETS)
    for override in args.npy_path:
        name, path = override.split("=", 1)
        npy_paths[name] = path

    with args.input.open() as f:
        all_results = json.load(f)

    results = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datasets": {},
    }
    rows = []

    for dataset_name, dataset_result in all_results.items():
        print(f"[{dataset_name}] loading", flush=True)
        train, queries, gt = load_dataset(dataset_result, args.data_dir, npy_paths)
        nprobes = [args.nprobe]
        results["datasets"][dataset_name] = {}

        for l_value in args.l_values:
            lire_key = f"SPFresh-L{l_value}"
            if lire_key in dataset_result["lire_l_sweep"]:
                add_row(rows, dataset_name, f"LIRE-{l_value}", "LIRE", l_value,
                        dataset_result["lire_l_sweep"][lire_key], args.nprobe)

            method = f"Local-CDKM-L{l_value}"
            print(f"[{dataset_name}] {method}", flush=True)
            idx = build_after_stream(
                L2LocalCDKMRepairNovaIVF,
                {"use_scdkm": True, "local_l": l_value, "local_iter": args.local_iter},
                train,
                int(dataset_result["nlist"]),
                int(dataset_result["n_init"]),
                int(dataset_result["n_stream"]),
                seed=int(dataset_result["seed"]),
            )
            summary = summarize_index(idx, queries, gt, nprobes, int(dataset_result["k"]))
            summary["local_l"] = int(l_value)
            summary["local_iter"] = int(args.local_iter)
            summary["n_local_candidates"] = int(getattr(idx, "n_local_candidates", 0))
            summary["local_cdkm_time_s"] = float(getattr(idx, "total_local_cdkm_time", 0.0))
            results["datasets"][dataset_name][method] = summary
            add_row(rows, dataset_name, method, "Local-CDKM", l_value, summary, args.nprobe)

    return results, rows


def write_outputs(results, rows, output, csv_path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, pdf_path, png_path):
    datasets = sorted({r["dataset"] for r in rows})
    l_values = sorted({int(r["L"]) for r in rows})
    metrics = [
        ("recall_at_nprobe", "Recall@100"),
        ("npa_violation_rate", "NPA violation"),
        ("maintenance_time_s", "Maintenance time (s)"),
        ("reassignments", "Moved vectors"),
    ]

    plt.rcParams.update({
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.1))
    axes = axes.ravel()
    colors = dict(zip(datasets, plt.cm.tab10.colors))
    markers = ["o", "s", "^", "D", "v"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for i, dataset in enumerate(datasets):
            for family, linestyle, label in [("LIRE", "--", dataset), ("Local-CDKM", "-", "_nolegend_")]:
                ds_rows = [r for r in rows if r["dataset"] == dataset and r["family"] == family]
                ds_rows.sort(key=lambda r: int(r["L"]))
                if not ds_rows:
                    continue
                ax.plot(
                    [int(r["L"]) for r in ds_rows],
                    [float(r[metric]) for r in ds_rows],
                    color=colors[dataset],
                    marker=markers[i % len(markers)],
                    linestyle=linestyle,
                    linewidth=1.25,
                    markersize=3.7,
                    label=label,
                )
        ax.set_xlabel("L")
        ax.set_ylabel(ylabel)
        ax.set_xticks(l_values)
        if metric in ("maintenance_time_s", "reassignments"):
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.35, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0, h_pad=0.9)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    if png_path is not None:
        fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/lire_l_sweep_5datasets_real_50k_with_nora.json"))
    parser.add_argument("--output", type=Path, default=Path("results/local_cdkm_repair_diagnostic_50k.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/local_cdkm_repair_diagnostic_50k.csv"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/local_cdkm_repair_diagnostic_50k.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/local_cdkm_repair_diagnostic_50k.png"))
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--npy-path", action="append", default=[])
    parser.add_argument("--l-values", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--local-iter", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=8)
    args = parser.parse_args()

    results, rows = run_diagnostic(args)
    write_outputs(results, rows, args.output, args.csv)
    plot_rows(rows, args.pdf, args.png)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
