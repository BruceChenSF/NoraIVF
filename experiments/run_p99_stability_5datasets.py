#!/usr/bin/env python3
"""Run the five-dataset P99 query-latency stability table."""

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
import os
from pathlib import Path

import numpy as np

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from experiments.run_claim_evidence_annbench import L2FrozenIVF, L2NovaIVF, L2SPFreshIVF
from experiments.run_experiments import StreamSimulator, run_experiment


DATASETS = [
    {
        "name": "mnist-784-euclidean",
        "label": "MNIST",
        "format": "hdf5",
        "path": "/Users/bruce/Downloads/mnist-784-euclidean.hdf5",
    },
    {
        "name": "fashion-mnist-784-euclidean",
        "label": "Fashion",
        "format": "hdf5",
        "path": "/Users/bruce/Downloads/fashion-mnist-784-euclidean.hdf5",
    },
    {
        "name": "core50",
        "label": "CoRE50",
        "format": "npy",
        "path": "data/core50",
    },
    {
        "name": "ScanNetCLIP",
        "label": "ScanNetCLIP",
        "format": "npy",
        "path": "data/ScanNetCLIP",
    },
    {
        "name": "Ego4D",
        "label": "Ego4D",
        "format": "npy",
        "path": "data/Ego4D",
    },
]

METHODS = [
    ("NoraIVF", L2NovaIVF, {"use_scdkm": True}),
    ("SPFresh", L2SPFreshIVF, {}),
    ("Frozen", L2FrozenIVF, {}),
]


def load_dataset(spec: dict, max_train: int, max_queries: int, k: int):
    if spec["format"] == "hdf5":
        return load_ann_benchmark_hdf5(
            spec["path"], max_train=max_train, max_queries=max_queries, k=k
        )
    if spec["format"] == "npy":
        return load_npy_vector_dataset_l2(
            spec["path"],
            max_train=max_train,
            max_queries=max_queries,
            k=k,
            recompute_gt=False,
        )
    raise ValueError(f"Unsupported dataset format: {spec['format']}")


def run_table(args: argparse.Namespace) -> dict:
    results = {
        "config": {
            "max_train": args.max_train,
            "max_queries": args.max_queries,
            "nlist": args.nlist,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "nprobe": args.nprobe,
            "k": args.k,
            "seed": args.seed,
        },
        "rows": [],
    }

    for spec in DATASETS:
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        print(f"[{spec['label']}] loading {path}", flush=True)
        data, queries, gt = load_dataset(spec, args.max_train, args.max_queries, args.k)
        n_init = min(len(data) // 2, args.n_init)
        stream = data[n_init:]

        row = {
            "dataset": spec["name"],
            "label": spec["label"],
        }
        for method_name, cls, kwargs in METHODS:
            print(f"[{spec['label']}] {method_name}", flush=True)
            index = cls(data[:n_init], args.nlist, seed=args.seed, **kwargs)
            simulator = StreamSimulator(stream, queries, gt, seed=args.seed)
            run = run_experiment(
                index,
                simulator,
                n_batches=args.n_batches,
                batch_size=args.batch_size,
                nprobe=args.nprobe,
                k=args.k,
                warmup_batches=args.warmup_batches,
            )
            row[method_name] = {
                "p99_ms": float(run["p99_latency"]),
                "p50_ms": float(run["p50_latency"]),
                "avg_qps": float(run["avg_qps"]),
                "avg_recall": float(run["avg_recall"]),
                "n_queries": int(len(run["query_latencies"])),
                "query_latencies_ms": [float(x) for x in run["query_latencies"]],
                "recall_trace": [float(x) for x in run["recall"]],
                "batch_qps": [float(x) for x in run["qps"][args.warmup_batches :]],
            }
            print(
                f"  P99={row[method_name]['p99_ms']:.3f}ms, "
                f"QPS={row[method_name]['avg_qps']:.1f}",
                flush=True,
            )
        results["rows"].append(row)
    return results


def write_outputs(results: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "noraivf_ms", "spfresh_ms", "frozen_ms"])
        for row in results["rows"]:
            writer.writerow(
                [
                    row["label"],
                    f"{row['NoraIVF']['p99_ms']:.3f}",
                    f"{row['SPFresh']['p99_ms']:.3f}",
                    f"{row['Frozen']['p99_ms']:.3f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--n-init", type=int, default=15000)
    parser.add_argument("--n-batches", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=Path("results/p99_stability_5datasets.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/p99_stability_5datasets.csv"))
    args = parser.parse_args()

    results = run_table(args)
    write_outputs(results, args.json, args.csv)
    print(f"Wrote {args.json}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
