#!/usr/bin/env python3
"""
Diagnostic experiment for boundary-sampled SCDKM.

The variant samples a small number of external boundary points from nearby
clusters and uses them as ghost statistics during SCDKM split optimization.  The
ghost points are never reassigned, so the experiment tests whether a small
amount of boundary signal can approximate LIRE's repair semantics without
explicit boundary migration.
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


class L2BoundarySampledNovaIVF(L2NovaIVF):
    def __init__(self, *args, boundary_neighbors=2, boundary_samples=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_neighbors = int(boundary_neighbors)
        self.boundary_samples = int(boundary_samples)
        self.n_boundary_ghosts = 0

    def _sample_boundary_ghosts(self, c1, c2):
        if self.boundary_neighbors <= 0 or self.boundary_samples <= 0:
            return []

        active = [c for c in range(self.k) if c not in (c1, c2) and len(self.posting_lists[c]) > 0]
        if not active:
            return []

        center = 0.5 * (self.centroids[c1] + self.centroids[c2])
        dists = [(float(squared_l2_to_centroids(center, self.centroids[c:c + 1])[0]), c) for c in active]
        dists.sort(key=lambda item: item[0])
        neighbors = [c for _, c in dists[: self.boundary_neighbors]]

        ghosts = []
        for c in neighbors:
            ids = list(self.posting_lists[c])
            if len(ids) <= self.boundary_samples:
                selected = ids
            else:
                scored = []
                for idx in ids:
                    vec = self.id_to_vec[idx]
                    d1 = squared_l2_to_centroids(vec, self.centroids[c1:c1 + 1])[0]
                    d2 = squared_l2_to_centroids(vec, self.centroids[c2:c2 + 1])[0]
                    scored.append((abs(float(d1 - d2)), min(float(d1), float(d2)), idx))
                scored.sort(key=lambda item: (item[0], item[1]))
                selected = [idx for _, _, idx in scored[: self.boundary_samples]]
            ghosts.extend(self.id_to_vec[idx] for idx in selected if idx in self.id_to_vec)
        self.n_boundary_ghosts += len(ghosts)
        return ghosts

    @staticmethod
    def _ghost_stats(ghosts, centroids, c1, c2):
        stats = {
            c1: [0, np.zeros(centroids.shape[1], dtype=np.float32)],
            c2: [0, np.zeros(centroids.shape[1], dtype=np.float32)],
        }
        for vec in ghosts:
            d1 = squared_l2_to_centroids(vec, centroids[c1:c1 + 1])[0]
            d2 = squared_l2_to_centroids(vec, centroids[c2:c2 + 1])[0]
            c = c1 if d1 <= d2 else c2
            stats[c][0] += 1
            stats[c][1] += vec
        out = {}
        for c, (n, a) in stats.items():
            out[c] = (n, a, float(np.dot(a, a)))
        return out

    def _virtual_cluster_stats(self, c, ghost_stats):
        gn, ga, _ = ghost_stats[c]
        n = float(self.n_j[c] + gn)
        a = self.a_j[c] + ga
        s = float(np.dot(a, a))
        return n, a, s

    def _cdkm_iterate(self, c1, c2, candidate_indices, n_iter=5):
        t0 = time.perf_counter()
        ghosts = self._sample_boundary_ghosts(c1, c2)
        if not ghosts:
            out = super()._cdkm_iterate(c1, c2, candidate_indices, n_iter=n_iter)
            return out

        for _ in range(n_iter):
            changed = False
            ghost_stats = self._ghost_stats(ghosts, self.centroids, c1, c2)
            for idx in list(candidate_indices):
                if idx not in self.id_to_vec or idx not in self.ppm:
                    continue
                p = self.ppm[idx]
                if p not in (c1, c2):
                    continue
                vec = self.id_to_vec[idx]
                rho_i = float(np.dot(vec, vec))

                best_c = p
                best_phi = -np.inf
                for c in (c1, c2):
                    n, a, s = self._virtual_cluster_stats(c, ghost_stats)
                    if c == p:
                        if self.n_j[c] <= 1 or n <= 1:
                            phi = 0.0
                        else:
                            s_reduced = s - 2.0 * float(np.dot(vec, a)) + rho_i
                            phi = s / n - s_reduced / (n - 1.0)
                    else:
                        s_augmented = s + 2.0 * float(np.dot(vec, a)) + rho_i
                        phi = s_augmented / (n + 1.0) - s / n if n > 0 else rho_i
                    if phi > best_phi:
                        best_phi = phi
                        best_c = c

                if best_c != p:
                    self._move_vector(idx, p, best_c, vec)
                    changed = True
            if not changed:
                break

        self.total_scdkm_time += time.perf_counter() - t0


class L2SampledRepairNovaIVF(L2BoundarySampledNovaIVF):
    def _sample_boundary_ids(self, c1, c2):
        if self.boundary_neighbors <= 0 or self.boundary_samples <= 0:
            return []

        active = [c for c in range(self.k) if c not in (c1, c2) and len(self.posting_lists[c]) > 0]
        if not active:
            return []

        center = 0.5 * (self.centroids[c1] + self.centroids[c2])
        dists = [(float(squared_l2_to_centroids(center, self.centroids[c:c + 1])[0]), c) for c in active]
        dists.sort(key=lambda item: item[0])
        neighbors = [c for _, c in dists[: self.boundary_neighbors]]

        sampled = []
        for c in neighbors:
            ids = list(self.posting_lists[c])
            if len(ids) <= self.boundary_samples:
                sampled.extend(ids)
                continue
            scored = []
            for idx in ids:
                vec = self.id_to_vec[idx]
                d1 = squared_l2_to_centroids(vec, self.centroids[c1:c1 + 1])[0]
                d2 = squared_l2_to_centroids(vec, self.centroids[c2:c2 + 1])[0]
                scored.append((abs(float(d1 - d2)), min(float(d1), float(d2)), idx))
            scored.sort(key=lambda item: (item[0], item[1]))
            sampled.extend(idx for _, _, idx in scored[: self.boundary_samples])
        self.n_boundary_ghosts += len(sampled)
        return sampled

    def _cdkm_iterate(self, c1, c2, candidate_indices, n_iter=5):
        t0 = time.perf_counter()
        external_ids = set(self._sample_boundary_ids(c1, c2))
        all_candidates = list(candidate_indices) + list(external_ids)

        for _ in range(n_iter):
            changed = False
            for idx in list(all_candidates):
                if idx not in self.id_to_vec or idx not in self.ppm:
                    continue
                p = self.ppm[idx]
                vec = self.id_to_vec[idx]
                rho_i = float(np.dot(vec, vec))

                if p in (c1, c2):
                    best_c = p
                    best_phi = -np.inf
                    for c in (c1, c2):
                        if c == p:
                            if self.n_j[c] <= 1:
                                phi = 0.0
                            else:
                                s_reduced = self.s_j[c] - 2.0 * float(np.dot(vec, self.a_j[c])) + rho_i
                                phi = self.s_j[c] / self.n_j[c] - s_reduced / (self.n_j[c] - 1)
                        else:
                            s_augmented = self.s_j[c] + 2.0 * float(np.dot(vec, self.a_j[c])) + rho_i
                            phi = s_augmented / (self.n_j[c] + 1) - self.s_j[c] / self.n_j[c]
                        if phi > best_phi:
                            best_phi = phi
                            best_c = c
                    if best_c != p:
                        self._move_vector(idx, p, best_c, vec)
                        changed = True
                elif idx in external_ids:
                    best_c = p
                    best_phi = 0.0
                    for c in (c1, c2):
                        if self.n_j[c] <= 0:
                            phi = rho_i
                        else:
                            s_augmented = self.s_j[c] + 2.0 * float(np.dot(vec, self.a_j[c])) + rho_i
                            phi = s_augmented / (self.n_j[c] + 1) - self.s_j[c] / self.n_j[c]
                        if phi > best_phi:
                            best_phi = phi
                            best_c = c
                    if best_c != p:
                        self._move_vector(idx, p, best_c, vec)
                        self.n_reassigns += 1
                        changed = True
            if not changed:
                break

        self.total_scdkm_time += time.perf_counter() - t0


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
    raise ValueError(f"Unsupported format: {dataset_result['format']}")


def recall_at_nprobe(result, nprobe):
    frontier = result["frontier"]
    nprobes = frontier["nprobe"]
    idx = nprobes.index(nprobe) if nprobe in nprobes else min(
        range(len(nprobes)), key=lambda i: abs(nprobes[i] - nprobe)
    )
    return float(frontier["recall"][idx])


def baseline_rows(dataset_name, dataset_result, nprobe):
    rows = []
    for method, source in [
        ("LIRE-0", dataset_result["lire_l_sweep"].get("SPFresh-L0")),
        ("LIRE-5", dataset_result["lire_l_sweep"].get("SPFresh-L5")),
        ("Nora", dataset_result.get("lire_l_sweep_baselines", {}).get("Nora")),
    ]:
        if source is None:
            continue
        rows.append({
            "dataset": dataset_name,
            "method": method,
            "sample_per_neighbor": None,
            "family": "baseline",
            "boundary_ghosts": None,
            "reassignments": int(source["n_reassigns"]),
            "maintenance_time_s": float(source["maintenance_time_s"]),
            "npa_violation_rate": float(source["npa"]["violation_rate"]),
            "recall_at_nprobe": recall_at_nprobe(source, nprobe),
        })
    return rows


def run_diagnostic(args):
    npy_paths = dict(NPY_DATASETS)
    for override in args.npy_path:
        name, path = override.split("=", 1)
        npy_paths[name] = path

    with args.input.open() as f:
        all_results = json.load(f)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    out = {"config": config, "datasets": {}}
    rows = []

    for dataset_name, dataset_result in all_results.items():
        print(f"[{dataset_name}] loading", flush=True)
        train, queries, gt = load_dataset(dataset_result, args.data_dir, npy_paths)
        nprobes = [args.nprobe]
        out["datasets"][dataset_name] = {"boundary_sampled": {}, "baselines": {}}
        rows.extend(baseline_rows(dataset_name, dataset_result, args.nprobe))

        for sample_count in args.samples:
            for method_prefix, family, index_cls in [
                ("BS-CDKM", "boundary_sampled", L2BoundarySampledNovaIVF),
                ("SR-CDKM", "sampled_repair", L2SampledRepairNovaIVF),
            ]:
                if args.mode != "both" and args.mode != family:
                    continue
                method = f"{method_prefix}-s{sample_count}"
                print(f"[{dataset_name}] {method}", flush=True)
                idx = build_after_stream(
                    index_cls,
                    {
                        "use_scdkm": True,
                        "boundary_neighbors": args.neighbors,
                        "boundary_samples": sample_count,
                    },
                    train,
                    int(dataset_result["nlist"]),
                    int(dataset_result["n_init"]),
                    int(dataset_result["n_stream"]),
                    seed=int(dataset_result["seed"]),
                )
                summary = summarize_index(idx, queries, gt, nprobes, int(dataset_result["k"]))
                summary["boundary_neighbors"] = int(args.neighbors)
                summary["boundary_samples"] = int(sample_count)
                summary["n_boundary_ghosts"] = int(getattr(idx, "n_boundary_ghosts", 0))
                out["datasets"][dataset_name]["boundary_sampled"][method] = summary
                rows.append({
                    "dataset": dataset_name,
                    "method": method,
                    "sample_per_neighbor": int(sample_count),
                    "family": family,
                    "boundary_ghosts": int(getattr(idx, "n_boundary_ghosts", 0)),
                    "reassignments": int(summary["n_reassigns"]),
                    "maintenance_time_s": float(summary["maintenance_time_s"]),
                    "npa_violation_rate": float(summary["npa"]["violation_rate"]),
                    "recall_at_nprobe": recall_at_nprobe(summary, args.nprobe),
                })

    return out, rows


def write_outputs(out, rows, output, csv_path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(out, f, indent=2)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, pdf_path, png_path):
    datasets = sorted({r["dataset"] for r in rows})
    sample_values = sorted({int(r["sample_per_neighbor"]) for r in rows if r["family"] == "boundary_sampled"})
    metrics = [
        ("recall_at_nprobe", "Recall@100"),
        ("npa_violation_rate", "NPA violation"),
        ("maintenance_time_s", "Maintenance time (s)"),
    ]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    colors = dict(zip(datasets, plt.cm.tab10.colors))
    markers = ["o", "s", "^", "D", "v"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for i, dataset in enumerate(datasets):
            ds = [r for r in rows if r["dataset"] == dataset]
            bs = [r for r in ds if r["family"] in ("boundary_sampled", "sampled_repair")]
            bs.sort(key=lambda r: int(r["sample_per_neighbor"]))
            for family, linestyle in [("boundary_sampled", "-"), ("sampled_repair", "--")]:
                fam_rows = [r for r in bs if r["family"] == family]
                if not fam_rows:
                    continue
                ax.plot(
                    [int(r["sample_per_neighbor"]) for r in fam_rows],
                    [float(r[metric]) for r in fam_rows],
                    marker=markers[i % len(markers)],
                    color=colors[dataset],
                    linestyle=linestyle,
                    linewidth=1.35,
                    markersize=3.8,
                    label=dataset if family == "boundary_sampled" else "_nolegend_",
                )
            for baseline, style in [("Nora", ":"), ("LIRE-0", "--"), ("LIRE-5", "-.")]:
                row = next((r for r in ds if r["method"] == baseline), None)
                if row is not None:
                    ax.plot(sample_values, [float(row[metric])] * len(sample_values),
                            color=colors[dataset], linestyle=style, linewidth=0.8, alpha=0.45,
                            label="_nolegend_")
        ax.set_xlabel("Ghost samples / neighbor")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sample_values)
        if metric == "maintenance_time_s":
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.35, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.1)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    if png_path is not None:
        fig.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/lire_l_sweep_5datasets_real_50k_with_nora.json"))
    parser.add_argument("--output", type=Path, default=Path("results/boundary_sampled_scdkm_diagnostic_50k.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/boundary_sampled_scdkm_diagnostic_50k.csv"))
    parser.add_argument("--pdf", type=Path, default=Path("figures/boundary_sampled_scdkm_diagnostic_50k.pdf"))
    parser.add_argument("--png", type=Path, default=Path("figures/boundary_sampled_scdkm_diagnostic_50k.png"))
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--npy-path", action="append", default=[])
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--samples", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--mode", choices=["boundary_sampled", "sampled_repair", "both"], default="both")
    parser.add_argument("--nprobe", type=int, default=8)
    args = parser.parse_args()

    out, rows = run_diagnostic(args)
    write_outputs(out, rows, args.output, args.csv)
    plot_rows(rows, args.pdf, args.png)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.pdf}")
    print(f"Wrote {args.png}")


if __name__ == "__main__":
    main()
