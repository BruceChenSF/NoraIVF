"""Prepare local L2 vector datasets for the embodied-stream experiments.

The experiment runner expects each dataset directory to contain:
  train.npy, queries.npy, and gt.npy

These generated CLIP-style streams are unit-normalized, so Euclidean distance
and cosine similarity induce the same nearest-neighbor order.
"""
import sys
import argparse
import json
import os
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from noraivf.datasets import generate_embodied_dataset


PROFILES = {
    "core50": {
        "dirname": "core50",
        "seed": 1501,
        "n_clusters": 64,
        "cluster_std": 0.34,
        "drift_frac": 0.28,
    },
    "ScanNetCLIP": {
        "dirname": "ScanNetCLIP",
        "seed": 2602,
        "n_clusters": 80,
        "cluster_std": 0.42,
        "drift_frac": 0.36,
    },
    "Ego4D": {
        "dirname": "Ego4D",
        "seed": 3703,
        "n_clusters": 72,
        "cluster_std": 0.48,
        "drift_frac": 0.45,
    },
}


def write_dataset(name, profile, out_root, n_vectors, n_queries, dim, overwrite):
    out_dir = os.path.join(out_root, profile["dirname"])
    train_path = os.path.join(out_dir, "train.npy")
    queries_path = os.path.join(out_dir, "queries.npy")
    gt_path = os.path.join(out_dir, "gt.npy")

    if not overwrite and all(os.path.exists(p) for p in (train_path, queries_path, gt_path)):
        print(f"[{name}] exists, skipping {out_dir}", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] generating {n_vectors} vectors, {n_queries} queries, d={dim}", flush=True)
    train, queries, gt = generate_embodied_dataset(
        n_vectors=n_vectors,
        n_queries=n_queries,
        d=dim,
        n_clusters=profile["n_clusters"],
        cluster_std=profile["cluster_std"],
        drift_frac=profile["drift_frac"],
        seed=profile["seed"],
    )

    np.save(train_path, train.astype(np.float32))
    np.save(queries_path, queries.astype(np.float32))
    np.save(gt_path, gt.astype(np.int32))

    metadata = {
        "name": name,
        "kind": "generated_unit_normalized_l2_stream",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_vectors": int(train.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dimension": int(train.shape[1]),
        "gt_k": int(gt.shape[1]),
        "generator": "noraivf/datasets.py:generate_embodied_dataset",
        "profile": {k: v for k, v in profile.items() if k != "dirname"},
        "distance": "squared Euclidean on unit-normalized vectors",
        "note": (
            "Generated local CLIP-style embodied stream used to rerun LIRE "
            "neighborhood-size sensitivity when original raw feature files "
            "are not present in the workspace."
        ),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[{name}] wrote {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data")
    parser.add_argument("--datasets", nargs="+", default=list(PROFILES.keys()))
    parser.add_argument("--n-vectors", type=int, default=30000)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in args.datasets:
        if name not in PROFILES:
            raise KeyError(f"Unknown dataset {name}. Known: {sorted(PROFILES)}")
        write_dataset(
            name,
            PROFILES[name],
            args.out_root,
            args.n_vectors,
            args.n_queries,
            args.dim,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
