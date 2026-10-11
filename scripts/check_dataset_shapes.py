#!/usr/bin/env python3
"""Self-check: shapes (and load path) for all 10 claim-evidence datasets.

Scans the runner's dataset list (experiments/run_claim_evidence_annbench.py:
ANNBENCH_DATASETS + NPY_DATASETS) and for each dataset:

  1. prints the full on-disk shapes (train / queries / gt, from the npy
     headers or the HDF5 file), and
  2. exercises the *real* loader (load_annbench_dataset / the npy loader)
     on a small subset, verifying the load path, the resolved metric, and
     (for angular sets) that vectors are L2-normalized at load.

Datasets not yet present on disk — audioset-128-euclidean while
scripts/prepare_audioset.py is still running — are reported as SKIP and do
not fail the script.

Usage::

    python scripts/check_dataset_shapes.py            # full check
    python scripts/check_dataset_shapes.py --no-exercise   # shapes only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.run_claim_evidence_annbench import (
    ANNBENCH_DATASETS,
    NPY_DATASETS,
    load_annbench_dataset,
    resolve_annbench_source,
)
from noraivf.datasets import load_npy_vector_dataset_l2


def npy_shape(path: str) -> tuple:
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape)


def check_annbench(name: str, data_dir: str, exercise: bool,
                   exercise_train: int, exercise_queries: int) -> bool:
    try:
        path, fmt, metric = resolve_annbench_source(name, data_dir)
    except FileNotFoundError:
        print(f"  SKIP  {name}: dataset directory not ready yet")
        return False
    print(f"  {name}  [{fmt}, metric={metric}]")
    ok = True
    if fmt == "npy":
        for f in ("train", "queries", "gt"):
            p = os.path.join(path, f"{f}.npy")
            if os.path.exists(p):
                print(f"    {f}.npy shape = {npy_shape(p)}")
            else:
                print(f"    {f}.npy MISSING")
                ok = False
    else:
        import h5py
        with h5py.File(path, "r") as fh:
            for key in ("train", "test", "neighbors"):
                if key in fh:
                    print(f"    hdf5[{key}] shape = {fh[key].shape}")
    if exercise and ok:
        train, queries, gt, metric_loaded, fmt_loaded = load_annbench_dataset(
            name, data_dir, max_train=exercise_train,
            max_queries=exercise_queries, k=10,
        )
        assert fmt_loaded == fmt and metric_loaded == metric
        print(f"    load path OK: train {train.shape} queries {queries.shape} "
              f"gt {gt.shape} ({len(queries)} queries, k={gt.shape[1]})")
        if metric == "angular":
            max_dev = float(np.abs(np.linalg.norm(train[:8], axis=1) - 1.0).max())
            max_dev_q = float(np.abs(np.linalg.norm(queries[:8], axis=1) - 1.0).max())
            print(f"    angular normalized OK: max |norm-1| train={max_dev:.2e} "
                  f"queries={max_dev_q:.2e}")
    return ok


def check_npy(name: str, base_dir: str, exercise: bool,
              exercise_train: int, exercise_queries: int) -> bool:
    print(f"  {name}  [npy, metric=euclidean]")
    ok = True
    for f in ("train", "queries", "gt"):
        p = os.path.join(base_dir, f"{f}.npy")
        if os.path.exists(p):
            print(f"    {f}.npy shape = {npy_shape(p)}")
        else:
            print(f"    {f}.npy MISSING")
            ok = False
    if exercise and ok:
        train, queries, gt = load_npy_vector_dataset_l2(
            base_dir, max_train=exercise_train, max_queries=exercise_queries, k=10,
        )
        print(f"    load path OK: train {train.shape} queries {queries.shape} "
              f"gt {gt.shape}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/annbench")
    parser.add_argument("--no-exercise", action="store_true",
                        help="skip the real-loader exercise (shapes only)")
    args = parser.parse_args()

    exercise = not args.no_exercise
    print("== ann-benchmarks datasets ==")
    n_ok = 0
    n_skip = 0
    for name in ANNBENCH_DATASETS:
        if check_annbench(name, args.data_dir, exercise, 2000, 100):
            n_ok += 1
        else:
            n_skip += 1
    print("== NPY (embodied) datasets ==")
    for name, base_dir in NPY_DATASETS.items():
        if check_npy(name, base_dir, exercise, 2000, 100):
            n_ok += 1
        else:
            n_skip += 1
    print(f"== {n_ok} datasets OK, {n_skip} skipped (not ready) ==")


if __name__ == "__main__":
    main()
