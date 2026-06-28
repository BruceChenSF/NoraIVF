#!/usr/bin/env python3
"""Append NoraIVF baselines to an existing LIRE L-sweep result JSON."""

from __future__ import annotations

import sys
import argparse
import json
import os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from experiments.run_claim_evidence_annbench import (
    DATASET_URLS,
    NPY_DATASETS,
    L2NovaIVF,
    build_after_stream,
    summarize_index,
)


def load_dataset(dataset_result: dict, data_dir: str, npy_paths: dict[str, str]):
    name = dataset_result["dataset"]
    k = int(dataset_result["k"])
    max_train = int(dataset_result["n_train"])
    max_queries = int(dataset_result["n_queries"])
    if dataset_result["format"] == "hdf5":
        path = os.path.join(data_dir, f"{name}.hdf5")
        return load_ann_benchmark_hdf5(path, max_train=max_train, max_queries=max_queries, k=k)
    if dataset_result["format"] == "npy":
        path = npy_paths[name]
        return load_npy_vector_dataset_l2(path, max_train=max_train, max_queries=max_queries, k=k)
    raise ValueError(f"Unsupported dataset format: {dataset_result['format']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--npy-path", action="append", default=[],
                        help="Override NPY dataset path as NAME=PATH.")
    args = parser.parse_args()

    npy_paths = dict(NPY_DATASETS)
    for override in args.npy_path:
        if "=" not in override:
            raise ValueError(f"--npy-path expects NAME=PATH, got {override}")
        name, path = override.split("=", 1)
        npy_paths[name] = path

    with args.input.open() as f:
        all_results = json.load(f)

    for dataset_name, dataset_result in all_results.items():
        if "lire_l_sweep_baselines" in dataset_result and "Nora" in dataset_result["lire_l_sweep_baselines"]:
            print(f"[{dataset_name}] Nora already present; skipping", flush=True)
            continue
        print(f"[{dataset_name}] Nora: build+stream", flush=True)
        train, queries, gt = load_dataset(dataset_result, args.data_dir, npy_paths)
        nprobes = next(iter(dataset_result["lire_l_sweep"].values()))["frontier"]["nprobe"]
        idx = build_after_stream(
            L2NovaIVF,
            {"use_scdkm": True},
            train,
            int(dataset_result["nlist"]),
            int(dataset_result["n_init"]),
            int(dataset_result["n_stream"]),
            seed=int(dataset_result["seed"]),
        )
        dataset_result.setdefault("lire_l_sweep_baselines", {})["Nora"] = summarize_index(
            idx,
            queries,
            gt,
            nprobes,
            int(dataset_result["k"]),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
