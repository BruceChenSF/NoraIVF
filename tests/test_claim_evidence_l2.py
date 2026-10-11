import sys
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.run_claim_evidence_annbench import (
    METHODS,
    L2SPFreshIVF,
    build_after_stream,
    compute_gt_on_corpus,
    count_npa_violations_l2,
    qps_at_recall,
    run_maintenance_budget,
)

from noraivf.datasets import compute_l2_ground_truth


class ToyIndex:
    def __init__(self):
        self.centroids = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        self.posting_lists = [[0, 1], []]
        self.id_to_vec = {
            0: np.array([0.1, 0.0], dtype=np.float32),
            1: np.array([9.9, 0.0], dtype=np.float32),
        }


def test_count_npa_violations_l2_detects_wrong_partition():
    stats = count_npa_violations_l2(ToyIndex())

    assert stats["checked"] == 2
    assert stats["violations"] == 1
    assert stats["violation_rate"] == 0.5


def test_l2_spfresh_accepts_lire_neighborhood_size():
    data = np.array(
        [[0, 0], [0, 1], [10, 0], [10, 1], [20, 0], [20, 1]],
        dtype=np.float32,
    )

    idx = L2SPFreshIVF(data, nlist=2, L=1)

    assert idx.L == 1


def test_main_l2_runner_exposes_six_dynamic_baselines():
    assert list(METHODS) == ["Frozen", "Rebuild", "DeDrift", "SPFresh", "NovaIVF", "AdaIVF"]


def test_all_main_l2_baselines_can_stream_and_search_toy_data():
    rng = np.random.RandomState(0)
    data = rng.randn(80, 6).astype(np.float32)

    for _, (cls, kwargs) in METHODS.items():
        idx = build_after_stream(cls, kwargs, data, nlist=4, n_init=40, n_stream=20)
        result = idx.search(data[0], k=5, nprobe=2)
        assert len(result) > 0


def test_run_maintenance_budget_picks_qps_at_recall_winners():
    frontier_cheap = {"recall": [0.5, 0.8], "qps": [50, 20]}
    assert qps_at_recall(frontier_cheap, 0.5) == 50
    assert qps_at_recall(frontier_cheap, 0.8) == 20
    assert qps_at_recall(frontier_cheap, 0.9) is None

    results = {
        "cheap": {"maintenance_time_s": 0.1, "frontier": frontier_cheap},
        "best": {"maintenance_time_s": 0.2, "frontier": {"recall": [0.9], "qps": [10]}},
        "too_expensive": {"maintenance_time_s": 4000.0, "frontier": {"recall": [0.95], "qps": [100]}},
    }

    budget = run_maintenance_budget(results, budgets_hours=[1.0])["1.0"]

    assert budget["budget_hours"] == 1.0
    assert budget["budget_s"] == 3600.0
    assert budget["eligible"] == ["best", "cheap"]
    assert budget["by_recall"]["0.5"]["method"] == "cheap"
    assert budget["by_recall"]["0.7"]["method"] == "cheap"
    assert budget["by_recall"]["0.9"]["method"] == "best"


def test_compute_gt_on_corpus_restricts_neighbors_to_indexed_half():
    """Indexed-scope GT must only ever name vectors inside train[:corpus_end].

    Synthetic setup: the first half is the corpus the index covers (moderately
    spread); the second half holds near-duplicates of each query, so it would
    dominate the full-corpus GT but is *not* in the index.  An indexed GT whose
    neighbors fall outside the first half is therefore provably wrong.
    """
    rng = np.random.RandomState(7)
    n_indexed = 40
    n_queries = 5
    k = 10
    queries = rng.randn(n_queries, 8).astype(np.float32)
    corpus = (rng.randn(n_indexed, 8).astype(np.float32) * 2.0)
    tail = np.repeat(
        queries + 1e-3 * rng.randn(n_queries, 8).astype(np.float32), 2, axis=0
    )
    train = np.vstack([corpus, tail]).astype(np.float32)

    # sanity: the full-corpus GT is contaminated by the tail (not in the index)
    full_gt = compute_l2_ground_truth(train, queries, k=k)
    assert np.any(full_gt >= n_indexed)

    # small corpus_chunk forces the trailing-chunk (< k rows) merge path
    indexed_gt = compute_gt_on_corpus(
        train, queries, k, "euclidean", corpus_end=n_indexed,
        query_chunk=3, corpus_chunk=17,
    )
    assert indexed_gt.shape == (n_queries, k)
    assert indexed_gt.dtype == np.int32
    assert np.all(indexed_gt < n_indexed)  # never names the out-of-index tail
    # and must equal an independent exact computation restricted to the corpus
    exact = compute_l2_ground_truth(train[:n_indexed], queries, k=k)
    assert np.array_equal(indexed_gt, exact)
    # recomputing the scope actually changed the GT vs the full-corpus one
    assert not np.array_equal(full_gt, indexed_gt)

    # angular path: on L2-normalized vectors IP ordering == squared-L2 ordering
    train_n = train / np.linalg.norm(train, axis=1, keepdims=True)
    q_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    indexed_gt_ang = compute_gt_on_corpus(
        train_n, q_n, k, "angular", corpus_end=n_indexed,
        query_chunk=3, corpus_chunk=17,
    )
    assert np.all(indexed_gt_ang < n_indexed)
    # independent check: squared-L2 on normalized vectors == IP ordering
    assert np.array_equal(
        indexed_gt_ang,
        compute_l2_ground_truth(train_n[:n_indexed], q_n, k=k),
    )



if __name__ == "__main__":
    test_count_npa_violations_l2_detects_wrong_partition()
    test_l2_spfresh_accepts_lire_neighborhood_size()
    test_main_l2_runner_exposes_six_dynamic_baselines()
    test_all_main_l2_baselines_can_stream_and_search_toy_data()
    test_run_maintenance_budget_picks_qps_at_recall_winners()
    test_compute_gt_on_corpus_restricts_neighbors_to_indexed_half()
    print("claim evidence L2 tests passed")
