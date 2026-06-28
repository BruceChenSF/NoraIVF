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
    count_npa_violations_l2,
    select_best_under_budget,
)


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


def test_main_l2_runner_exposes_five_dynamic_baselines():
    assert list(METHODS) == ["Frozen", "Rebuild", "DeDrift", "SPFresh", "NovaIVF"]


def test_all_main_l2_baselines_can_stream_and_search_toy_data():
    rng = np.random.RandomState(0)
    data = rng.randn(80, 6).astype(np.float32)

    for _, (cls, kwargs) in METHODS.items():
        idx = build_after_stream(cls, kwargs, data, nlist=4, n_init=40, n_stream=20)
        result = idx.search(data[0], k=5, nprobe=2)
        assert len(result) > 0


def test_select_best_under_budget_prefers_highest_recall_within_budget():
    candidates = {
        "cheap": {"maintenance_time_s": 0.1, "frontier": {"recall": [0.7], "qps": [10]}},
        "best": {"maintenance_time_s": 0.2, "frontier": {"recall": [0.9], "qps": [8]}},
        "too_expensive": {"maintenance_time_s": 0.5, "frontier": {"recall": [0.95], "qps": [7]}},
    }

    selected = select_best_under_budget(candidates, budget_s=0.25, nprobe_index=0)

    assert selected["method"] == "best"
    assert selected["recall"] == 0.9


if __name__ == "__main__":
    test_count_npa_violations_l2_detects_wrong_partition()
    test_l2_spfresh_accepts_lire_neighborhood_size()
    test_main_l2_runner_exposes_five_dynamic_baselines()
    test_all_main_l2_baselines_can_stream_and_search_toy_data()
    test_select_best_under_budget_prefers_highest_recall_within_budget()
    print("claim evidence L2 tests passed")
