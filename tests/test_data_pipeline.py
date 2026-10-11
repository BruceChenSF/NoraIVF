"""Unit tests for the ann-benchmarks data pipeline
(run: python tests/test_data_pipeline.py)."""

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np

from noraivf import list_datasets
from noraivf.datasets import load_npy_vector_dataset_l2
from scripts.prepare_annbench_datasets import convert_dataset, _dataset_up_to_date


def _make_source_hdf5(path: Path, n=2000, d=16, q=50, k=100):
    rng = np.random.RandomState(0)
    train = rng.randn(n, d).astype(np.float32)
    queries = rng.randn(q, d).astype(np.float32)
    # exact ground truth via brute force
    dists = np.linalg.norm(train[None, :, :] - queries[:, None, :], axis=2)
    neighbors = np.argsort(dists, axis=1)[:, :k].astype(np.int32)
    distances = np.sort(dists, axis=1)[:, :k].astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=queries)
        f.create_dataset("neighbors", data=neighbors)
        f.create_dataset("distances", data=distances)
    return train, queries, neighbors, distances


def test_convert_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "toy-16-euclidean.hdf5"
        train, queries, neighbors, distances = _make_source_hdf5(src)
        out_root = tmp / "out"

        entry = convert_dataset(src, out_root)

        assert entry["name"] == "toy-16-euclidean"
        assert entry["n_vectors"] == 2000 and entry["dim"] == 16
        out_dir = out_root / "toy-16-euclidean"
        for fname in ("train.npy", "queries.npy", "gt.npy", "distances.npy", "meta.json"):
            assert (out_dir / fname).exists()

        # reload through the standard loader
        t, q, gt = load_npy_vector_dataset_l2(out_dir, k=10)
        assert t.shape == (2000, 16)
        assert q.shape == (50, 16)
        assert gt.shape == (50, 10)
        assert (gt < 2000).all()

        # npy contents match source exactly
        assert np.array_equal(np.load(out_dir / "train.npy"), train)
        assert np.array_equal(np.load(out_dir / "gt.npy"), neighbors)
        assert np.array_equal(np.load(out_dir / "distances.npy"), distances)

        # meta.json records provenance
        meta = json.loads((out_dir / "meta.json").read_text())
        assert meta["source"]["digest"]
        assert meta["shape"]["train"] == [2000, 16]
        assert meta["normalized"] is False


def test_idempotent_skip_and_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "toy-16-euclidean.hdf5"
        _make_source_hdf5(src)
        out_root = tmp / "out"

        first = convert_dataset(src, out_root)
        assert first is not None
        assert _dataset_up_to_date(out_root / "toy-16-euclidean", src)

        # second run skips (returns manifest info, does not rewrite)
        mtime_before = (out_root / "toy-16-euclidean" / "train.npy").stat().st_mtime_ns
        second = convert_dataset(src, out_root)
        assert second is not None
        mtime_after = (out_root / "toy-16-euclidean" / "train.npy").stat().st_mtime_ns
        assert mtime_before == mtime_after

        # overwrite forces re-conversion
        mtime_before = (out_root / "toy-16-euclidean" / "train.npy").stat().st_mtime_ns
        convert_dataset(src, out_root, overwrite=True)
        mtime_after = (out_root / "toy-16-euclidean" / "train.npy").stat().st_mtime_ns
        assert mtime_before != mtime_after


def test_gt_range_guard():
    """GT indices pointing outside train must be rejected (never silent corruption)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "bad-16-euclidean.hdf5"
        train, queries, neighbors, distances = _make_source_hdf5(src, n=100, q=10)
        neighbors[:] = 99999  # corrupt: indices beyond train size
        with h5py.File(src, "w") as f:
            f.create_dataset("train", data=train)
            f.create_dataset("test", data=queries)
            f.create_dataset("neighbors", data=neighbors)
            f.create_dataset("distances", data=distances)

        try:
            convert_dataset(src, tmp / "out")
            raised = False
        except AssertionError:
            raised = True
        assert raised


def test_list_datasets_discovers_converted_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "toy-16-euclidean.hdf5"
        _make_source_hdf5(src)
        convert_dataset(src, tmp / "out")

        found = list_datasets(tmp / "out")
        assert len(found) == 1
        assert found[0]["name"] == "toy-16-euclidean"
        assert found[0]["n_vectors"] == 2000
        assert found[0]["dim"] == 16
        assert found[0]["meta"]["normalized"] is False


if __name__ == "__main__":
    test_convert_and_reload()
    test_idempotent_skip_and_overwrite()
    test_gt_range_guard()
    test_list_datasets_discovers_converted_dirs()
    print("data pipeline tests passed")
