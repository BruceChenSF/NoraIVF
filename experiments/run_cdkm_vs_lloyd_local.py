#!/usr/bin/env python3
"""Compare finite-budget CDKM and Lloyd on local real-data k-means splits.

This is a diagnostic experiment. It samples real IVF partitions, runs k=2 local
clustering from the same initialization, and records the original k-means SSE.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from noraivf.core import kmeans_faiss


DATASETS = {
    "MNIST": ("hdf5", "/Users/bruce/Downloads/mnist-784-euclidean.hdf5"),
    "Fashion": ("hdf5", "/Users/bruce/Downloads/fashion-mnist-784-euclidean.hdf5"),
    "CoRE50": ("npy", "data/core50"),
    "ScanNet": ("npy", "data/ScanNetCLIP"),
    "Ego4D": ("npy", "data/Ego4D"),
}


def load_vectors(kind: str, path: str, max_train: int) -> np.ndarray:
    if kind == "hdf5":
        train, _, _ = load_ann_benchmark_hdf5(path, max_train=max_train, max_queries=1, k=1)
    else:
        train, _, _ = load_npy_vector_dataset_l2(path, max_train=max_train, max_queries=1, k=1)
    return np.asarray(train, dtype=np.float32)


def sse(x: np.ndarray, labels: np.ndarray, k: int = 2) -> float:
    total = float(np.sum(x * x))
    explained = 0.0
    for c in range(k):
        members = x[labels == c]
        if len(members) == 0:
            continue
        sums = members.sum(axis=0, dtype=np.float64)
        explained += float(np.dot(sums, sums) / len(members))
    return total - explained


def balanced_init(n: int, rng: np.random.RandomState) -> np.ndarray:
    labels = np.zeros(n, dtype=np.int32)
    labels[n // 2 :] = 1
    rng.shuffle(labels)
    return labels


def lloyd_curve(x: np.ndarray, init_labels: np.ndarray, t_values: list[int]) -> dict[int, float]:
    labels = init_labels.copy()
    out = {0: sse(x, labels)}
    max_t = max(t_values)
    for t in range(1, max_t + 1):
        centroids = []
        for c in range(2):
            members = x[labels == c]
            if len(members) == 0:
                centroids.append(x[np.argmax(np.sum((x - x.mean(axis=0)) ** 2, axis=1))])
            else:
                centroids.append(members.mean(axis=0))
        centroids = np.asarray(centroids, dtype=np.float32)
        d0 = np.sum((x - centroids[0]) ** 2, axis=1)
        d1 = np.sum((x - centroids[1]) ** 2, axis=1)
        labels = (d1 < d0).astype(np.int32)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = init_labels.copy()
        if t in t_values:
            out[t] = sse(x, labels)
    return out


def cdkm_curve(x: np.ndarray, init_labels: np.ndarray, t_values: list[int]) -> dict[int, float]:
    labels = init_labels.copy()
    n = np.array([(labels == c).sum() for c in range(2)], dtype=np.int64)
    sums = np.vstack([x[labels == c].sum(axis=0, dtype=np.float64) for c in range(2)])
    out = {0: sse(x, labels)}
    max_t = max(t_values)
    for t in range(1, max_t + 1):
        changed = False
        for i, vec in enumerate(x):
            src = int(labels[i])
            dst = 1 - src
            if n[src] <= 1:
                continue
            old_term = np.dot(sums[src], sums[src]) / n[src] + np.dot(sums[dst], sums[dst]) / n[dst]
            src_sum = sums[src] - vec
            dst_sum = sums[dst] + vec
            new_term = np.dot(src_sum, src_sum) / (n[src] - 1) + np.dot(dst_sum, dst_sum) / (n[dst] + 1)
            if new_term > old_term + 1e-9:
                sums[src] = src_sum
                sums[dst] = dst_sum
                n[src] -= 1
                n[dst] += 1
                labels[i] = dst
                changed = True
        if t in t_values:
            out[t] = sse(x, labels)
        if not changed:
            final_sse = sse(x, labels)
            for tt in t_values:
                if tt > t and tt not in out:
                    out[tt] = final_sse
            break
    return out


def sample_local_partitions(
    data: np.ndarray,
    nlist: int,
    n_parts: int,
    local_size: int,
    seed: int,
) -> list[np.ndarray]:
    _, assignments = kmeans_faiss(data, nlist, niter=20, seed=seed)
    rng = np.random.RandomState(seed)
    candidates = []
    for c in range(nlist):
        idx = np.flatnonzero(assignments == c)
        if len(idx) >= max(8, local_size // 2):
            candidates.append(idx)
    rng.shuffle(candidates)
    parts = []
    for idx in candidates[:n_parts]:
        if len(idx) > local_size:
            idx = rng.choice(idx, size=local_size, replace=False)
        parts.append(np.asarray(data[idx], dtype=np.float32))
    return parts


def run_dataset(name: str, kind: str, path: str, args) -> list[dict]:
    data = load_vectors(kind, path, args.max_train)
    print(f"[{name}] loaded {data.shape}", flush=True)
    parts = sample_local_partitions(data, args.nlist, args.local_parts, args.local_size, args.seed)
    rows = []
    t_values = args.t_values
    for part_id, x in enumerate(parts):
        rng = np.random.RandomState(args.seed + 1009 * part_id)
        labels0 = balanced_init(len(x), rng)
        lloyd = lloyd_curve(x, labels0, t_values)
        cdkm = cdkm_curve(x, labels0, t_values)
        base = lloyd[0]
        for t in t_values:
            rows.append({
                "dataset": name,
                "partition": part_id,
                "n": len(x),
                "T": t,
                "method": "Lloyd",
                "sse": lloyd[t],
                "relative_sse": lloyd[t] / base if base > 0 else 1.0,
            })
            rows.append({
                "dataset": name,
                "partition": part_id,
                "n": len(x),
                "T": t,
                "method": "CDKM",
                "sse": cdkm[t],
                "relative_sse": cdkm[t] / base if base > 0 else 1.0,
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    datasets = sorted({r["dataset"] for r in rows})
    for dataset in datasets:
        for t in sorted({r["T"] for r in rows}):
            vals = {}
            for method in ["CDKM", "Lloyd"]:
                arr = np.asarray([
                    r["relative_sse"] for r in rows
                    if r["dataset"] == dataset and r["T"] == t and r["method"] == method
                ], dtype=np.float64)
                vals[method] = arr
            c = vals["CDKM"]
            l = vals["Lloyd"]
            out.append({
                "dataset": dataset,
                "T": t,
                "cdkm_mean_relative_sse": float(c.mean()),
                "lloyd_mean_relative_sse": float(l.mean()),
                "cdkm_better_fraction": float(np.mean(c < l)),
                "mean_relative_gap_cdkm_minus_lloyd": float(c.mean() - l.mean()),
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], output: Path) -> None:
    datasets = sorted({r["dataset"] for r in summary})
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.5, 2.25), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        rows = [r for r in summary if r["dataset"] == dataset]
        ts = [r["T"] for r in rows]
        ax.plot(ts, [r["cdkm_mean_relative_sse"] for r in rows], marker="o", label="CDKM")
        ax.plot(ts, [r["lloyd_mean_relative_sse"] for r in rows], marker="s", label="Lloyd")
        ax.set_title(dataset)
        ax.set_xlabel("T")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("Relative SSE")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--local-parts", type=int, default=20)
    parser.add_argument("--local-size", type=int, default=256)
    parser.add_argument("--t-values", nargs="+", type=int, default=[0, 1, 2, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-csv", type=Path, default=Path("results/cdkm_vs_lloyd_local_raw.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/cdkm_vs_lloyd_local_summary.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("results/cdkm_vs_lloyd_local_summary.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/cdkm_vs_lloyd_local_diagnostic.pdf"))
    args = parser.parse_args()

    all_rows = []
    for name in args.datasets:
        kind, path = DATASETS[name]
        all_rows.extend(run_dataset(name, kind, path, args))
    summary = summarize(all_rows)
    write_csv(args.raw_csv, all_rows)
    write_csv(args.summary_csv, summary)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as f:
        json.dump(summary, f, indent=2)
    plot_summary(summary, args.figure)
    print(f"Wrote {args.raw_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.summary_json}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
