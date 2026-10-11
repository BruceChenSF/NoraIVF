"""Tests for the AdaIVF baseline (arXiv:2411.00970 incremental IVF maintenance)."""
import sys
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from noraivf.core import AdaIVF, create_index


def _make_data(n_total=2500, nlist=16, d=32, seed=7):
    """Unit-normalized data with nlist well-separated Gaussian clusters."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(nlist, d).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    per = int(np.ceil(n_total / nlist))
    blocks = [centers[i] + 0.05 * rng.randn(per, d) for i in range(nlist)]
    data = np.vstack(blocks).astype(np.float32)[:n_total]
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


def _assert_pam_consistent(idx):
    """Every live id appears in exactly one posting list, and PAM agrees."""
    n_live = len(idx.id_to_vec)
    seen = {}
    for c in range(idx.k):
        for i in idx.posting_lists[c]:
            seen[i] = seen.get(i, 0) + 1
    assert len(seen) == n_live, (len(seen), n_live)
    for i, cnt in seen.items():
        assert cnt == 1, f"id {i} appears in {cnt} posting lists"
        assert i in idx.id_to_vec
        assert idx.pam.get(i) is not None
        assert i in idx.posting_lists[idx.pam[i]]


def _stream(idx, data, n_stream=500, n_delete=200):
    ids = [idx.insert(data[2000 + j]) for j in range(n_stream)]
    for vid in ids[:n_delete]:
        idx.delete(vid)
    return ids


def test_factory_registers_adaivf():
    data = _make_data(n_total=300)
    for name in ("adaivf", "ada-ivf"):
        idx = create_index(name, data, nlist=4, seed=0)
        assert isinstance(idx, AdaIVF)


def test_pam_consistency_and_centroids_finite():
    data = _make_data()
    idx = AdaIVF(data[:2000], nlist=16, tau_s=20, check_every=100, seed=0)
    _assert_pam_consistent(idx)
    assert np.all(np.isfinite(idx.centroids[:idx.k]))

    _stream(idx, data)
    _assert_pam_consistent(idx)
    assert np.all(np.isfinite(idx.centroids[:idx.k]))


def test_maintenance_triggers_splits_and_stats():
    data = _make_data()
    idx = AdaIVF(data[:2000], nlist=16, tau_s=20, check_every=100, seed=0)
    _stream(idx, data)

    assert idx.cumul_maintenance_time > 0.0
    assert idx.total_split_time > 0.0
    assert idx.n_splits > 0
    assert idx.ada_stats["n_local_reindex"] >= 1
    assert idx.ada_stats["n_violations"] >= 1
    assert idx.ada_stats["n_points_reclustered"] >= 1
    assert idx.ada_stats["n_global_rebuild"] == 0  # safety valve not hit


def test_search_returns_only_live_ids():
    data = _make_data()
    idx = AdaIVF(data[:2000], nlist=16, tau_s=20, check_every=100, seed=0)
    _stream(idx, data)

    for q in data[::37]:
        result = idx.search(q, k=20, nprobe=4)
        assert len(result) > 0
        assert all(i in idx.id_to_vec for i in result)


def test_temperature_never_below_one():
    data = _make_data()
    idx = AdaIVF(data[:2000], nlist=16, tau_s=20, check_every=100, seed=0)
    # no queries in the streaming phase -> T stays exactly 1.0
    _stream(idx, data)
    assert np.all(idx.T >= 1.0 - 1e-12)

    # interleaved queries: probed partitions heat, cooled ones floor at 1.0
    for q in data[::17]:
        idx.search(q, k=10, nprobe=8)
    assert np.all(idx.T >= 1.0 - 1e-12)
    assert np.any(idx.T > 1.0 + 1e-9)  # at least one partition heated


def test_temperature_affects_violation_scoring():
    data = _make_data()
    idx = AdaIVF(data[:2000], nlist=16, tau_s=20, check_every=100, seed=0)
    idx.T[:] = 1.0
    f_before = idx._detect_violators()
    # uniform heating multiplies every f score by 50 -> previous violators
    # stay violators and near-threshold partitions newly qualify
    idx.T[:] = 50.0
    f_after = idx._detect_violators()
    assert set(f_after.tolist()) >= set(f_before.tolist())
    assert len(f_after) > 0
    assert len(f_after) >= len(f_before)


if __name__ == "__main__":
    test_factory_registers_adaivf()
    test_pam_consistency_and_centroids_finite()
    test_maintenance_triggers_splits_and_stats()
    test_search_returns_only_live_ids()
    test_temperature_never_below_one()
    test_temperature_affects_violation_scoring()
    print("AdaIVF tests passed")
