#!/usr/bin/env python
"""Data pipeline: convert local ann-benchmarks HDF5 files into canonical
NoraIVF dataset directories under ``data/annbench/<name>/``.

Each dataset directory is self-contained and loadable by the existing
loaders (``load_npy_vector_dataset_l2`` / ``load_ann_benchmark_hdf5``)::

    data/annbench/<name>/
        train.npy        # float32 base vectors (N, d)
        queries.npy      # float32 query vectors (Q, d)
        gt.npy           # int32 exact ground-truth neighbor ids (Q, k)
        distances.npy    # float32 ground-truth distances (Q, k)
        meta.json        # provenance: source file, fingerprints, shapes, dtype

The pipeline itself is logged with the structured logger
(``ExperimentLogger``), so every conversion run is reproducible and its
data fingerprints are recorded (rebuttal-experiments-plan.md §4).

Usage::

    python scripts/prepare_annbench_datasets.py \
        --source-dir /path/to/ann_benchmark \
        --out-root data/annbench

    # convert a subset, overwrite existing outputs, no logging
    python scripts/prepare_annbench_datasets.py \
        --datasets sift-128-euclidean mnist-784-euclidean \
        --overwrite --no-log

The conversion is idempotent: existing, unmodified datasets are skipped
(their meta.json stores the source fingerprint; `--overwrite` forces a
re-conversion). For >10^7-vector scale, add more ann-benchmarks /
big-ann-benchmarks HDF5 sources into --source-dir and re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from noraivf.logging import ExperimentLogger, fingerprint_file

GT_K = 100  # ground-truth neighbor columns kept from the HDF5


def _source_fingerprint(source: Path) -> dict:
    return fingerprint_file(source)


def _dataset_up_to_date(out_dir: Path, source: Path) -> bool:
    """True if the dataset dir exists and its meta.json matches the source."""
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    source_meta = meta.get("source", {})
    return source_meta.get("path") == str(source.resolve()) and (
        source_meta.get("digest") == _source_fingerprint(source)["digest"]
    )


def convert_dataset(
    source: Path,
    out_root: Path,
    logger: ExperimentLogger | None = None,
    overwrite: bool = False,
    k: int = GT_K,
) -> dict | None:
    """Convert one ann-benchmarks HDF5 file; returns its manifest entry or None if skipped."""
    import h5py

    name = source.name.removesuffix(".hdf5")
    out_dir = out_root / name

    if not overwrite and _dataset_up_to_date(out_dir, source):
        msg = f"{name}: up to date, skipping (use --overwrite to force)"
        print(msg)
        if logger:
            logger.info(msg, phase=name)
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        return {
            "name": name,
            "n_vectors": meta["shape"]["train"][0],
            "dim": meta["shape"]["train"][1],
            "n_queries": meta["shape"]["queries"][0],
            "path": str(out_dir),
        }

    src_fp = _source_fingerprint(source)
    if logger:
        logger.phase_start(name, source=str(source), source_digest=src_fp["digest"])

    with h5py.File(source, "r") as f:
        train = np.asarray(f["train"][:], dtype=np.float32)
        queries = np.asarray(f["test"][:], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:, :k], dtype=np.int32)
        distances = (
            np.asarray(f["distances"][:, :k], dtype=np.float32)
            if "distances" in f
            else None
        )

    n, d = train.shape
    assert neighbors.shape[1] == k, f"{name}: expected {k} GT cols, got {neighbors.shape}"
    assert int(neighbors.max()) < n, f"{name}: GT indices out of range for train size"

    # ann-benchmarks convention: filenames ending in -angular store cosine/angular
    # ground truth (GT neighbor order is by cosine similarity). Everything else is
    # Euclidean (L2). Experiments must L2-normalize angular vectors before search
    # so that L2 ordering matches the cosine GT (see noraivf.datasets loader).
    metric = "angular" if name.endswith("-angular") else "l2"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "queries.npy", queries)
    np.save(out_dir / "gt.npy", neighbors)
    if distances is not None:
        np.save(out_dir / "distances.npy", distances)

    meta = {
        "name": name,
        "source": {
            "path": str(source.resolve()),
            "digest": src_fp["digest"],
            "fingerprint_mode": src_fp["mode"],
            "size_bytes": src_fp["size_bytes"],
        },
        "shape": {
            "train": list(train.shape),
            "queries": list(queries.shape),
            "gt": list(neighbors.shape),
        },
        "dtype": "float32",
        "metric": metric,
        "gt_basis": "cosine" if metric == "angular" else "l2",
        "normalized": False,  # ann-benchmarks sets are raw; angular sets are
        # normalized at load time in the experiment loader (L2 on normalized
        # vectors == cosine order, matching the GT)
        "gt_k": k,
        "files": {
            "train": "train.npy",
            "queries": "queries.npy",
            "gt": "gt.npy",
            "distances": "distances.npy" if distances is not None else None,
        },
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # validation: spot-check a few rows round-trip through npy
    for arr_name, arr in (("train", train), ("queries", queries), ("gt", neighbors)):
        reloaded = np.load(out_dir / f"{arr_name}.npy")
        assert reloaded.shape == arr.shape
        if arr.size:
            assert np.array_equal(reloaded[:3, :8], arr[:3, :8]), f"{name}: {arr_name} mismatch"

    if logger:
        logger.metric(
            name,
            n_vectors=int(n),
            dim=int(d),
            n_queries=int(len(queries)),
            gt_k=int(k),
            size_mb=float(sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e6),
            source_digest=src_fp["digest"],
        )
        for npy in sorted(out_dir.glob("*.npy")) + [meta_path]:
            logger.add_data_provenance(npy)
        logger.phase_end(name, metrics={"status": "converted"})

    print(f"{name}: {n} x {d} vectors, {len(queries)} queries -> {out_dir}")
    return {
        "name": name,
        "n_vectors": int(n),
        "dim": int(d),
        "n_queries": int(len(queries)),
        "path": str(out_dir),
        "source_digest": src_fp["digest"],
    }


def write_manifest(out_root: Path, entries: list[dict]) -> None:
    manifest = {"generated_by": "prepare_annbench_datasets.py", "datasets": entries}
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing *.hdf5 ann-benchmarks files")
    parser.add_argument("--out-root", type=Path, default=Path("data/annbench"),
                        help="output root (default: data/annbench)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="names (hdf5 basenames without .hdf5); default: all in source-dir")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-convert even if already up to date")
    parser.add_argument("--no-log", action="store_true",
                        help="disable structured logging (still writes per-dataset meta.json)")
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    args = parser.parse_args()

    source_dir = args.source_dir
    hdf5_files = sorted(source_dir.glob("*.hdf5"))
    if args.datasets:
        wanted = {f"{n}.hdf5" if not n.endswith(".hdf5") else n for n in args.datasets}
        hdf5_files = [p for p in hdf5_files if p.name in wanted]
    if not hdf5_files:
        raise SystemExit(f"no ann-benchmarks HDF5 files found in {source_dir}")

    config = {
        "source_dir": str(source_dir),
        "out_root": str(args.out_root),
        "datasets": [p.stem for p in hdf5_files],
        "gt_k": GT_K,
    }
    logger = None if args.no_log else ExperimentLogger(
        experiment_id="data-prep-annbench",
        config=config,
        out_dir=args.log_dir,
    )

    entries: list[dict] = []
    try:
        with logger or _nullcontext():
            for src in hdf5_files:
                entry = convert_dataset(
                    src, args.out_root, logger=logger, overwrite=args.overwrite, k=GT_K
                )
                if entry:
                    entries.append(entry)
        if entries:
            write_manifest(args.out_root, entries)
            print(f"manifest: {args.out_root / 'manifest.json'} "
                  f"({len(entries)} datasets, total {sum(e['n_vectors'] for e in entries)} vectors)")
    except Exception as exc:  # noqa: BLE001 - surface with log context
        if logger:
            logger.error(exc, phase="pipeline")
        raise


class _nullcontext:
    """Minimal null-context fallback (Python 3.7+ contextlib.nullcontext exists; keep local)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
