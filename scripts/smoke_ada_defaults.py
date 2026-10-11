"""Smoke test: AdaIVF vs NovaIVF on real SIFT with default hyperparameters.

Scenario (from the bug report):
  - tr = first 50000 SIFT vectors, L2-normalized
  - qn = first 100 queries, L2-normalized
  - build index (nlist=256) on tr
  - insert tr[0:5000] (duplicate vectors -> new ids 50000..54999)
  - delete ids 0..1999
  - query 50 qn with k=10, nprobe=8

Recall: GT = brute-force top-10 id set over ALL live vectors (assembled
from idx.id_to_vec); recall = |GT ∩ returned| / 10.
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import faiss

from noraivf.core import AdaIVF, NovaIVF


def _normalize(x):
    x = np.asarray(x, dtype=np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def brute_force_topk(queries, live_vecs, live_ids, k=10):
    """Brute-force top-k over live vectors; map faiss positions -> real ids."""
    index = faiss.IndexFlatIP(live_vecs.shape[1])
    index.add(live_vecs)
    _, I = index.search(queries, k)
    return np.array([[live_ids[i] for i in row] for row in I])


def run_one(name, idx_cls, tr, qn, nlist=256, k=10, nprobe=8,
            n_insert=5000, n_delete=2000, n_query=50, **kwargs):
    t0 = time.perf_counter()
    idx = idx_cls(tr, nlist, **kwargs)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    for j in range(n_insert):
        idx.insert(tr[j])  # duplicate vectors -> new ids 50000..54999
    t_insert_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    for vid in range(n_delete):
        idx.delete(vid)
    t_del_wall = time.perf_counter() - t0

    # GT over all live vectors
    live_ids = sorted(idx.id_to_vec.keys())
    live_vecs = np.array([idx.id_to_vec[i] for i in live_ids], dtype=np.float32)
    gt = brute_force_topk(qn[:n_query], live_vecs, live_ids, k=k)

    recalls = []
    t_search = 0.0
    for q, gt_ids in zip(qn[:n_query], gt):
        t0 = time.perf_counter()
        res = idx.search(q, k=k, nprobe=nprobe)
        t_search += time.perf_counter() - t0
        res_set = set(res)
        recalls.append(len(res_set & set(gt_ids.tolist())) / k)

    print(f"=== {name} ===")
    print(f"  build        : {t_build:.3f}s")
    print(f"  insert(5000) : wall {t_insert_wall:.3f}s, cumulative {idx.total_insert_time:.3f}s")
    print(f"  delete(2000) : wall {t_del_wall:.3f}s, cumulative {idx.total_delete_time:.3f}s")
    print(f"  maint        : {idx.cumul_maintenance_time:.3f}s (split {idx.total_split_time:.3f}s, merge {idx.total_merge_time:.3f}s)")
    print(f"  search(50)   : {t_search:.3f}s")
    print(f"  n_splits     : {idx.n_splits}")
    print(f"  n_merges     : {idx.n_merges}")
    print(f"  n_reassigns  : {idx.n_reassigns}")
    if hasattr(idx, "ada_stats"):
        print(f"  ada_stats    : {idx.ada_stats}")
    print(f"  recall@10    : mean {np.mean(recalls):.4f} (min {np.min(recalls):.3f}, max {np.max(recalls):.3f})")
    print(f"  live         : {len(live_ids)} (expect {len(tr) + n_insert - n_delete})")
    print()
    return idx


def main():
    data_dir = _ROOT / "data" / "annbench" / "sift-128-euclidean"
    train = np.load(data_dir / "train.npy")
    queries = np.load(data_dir / "queries.npy")

    tr = _normalize(train[:50000])
    qn = _normalize(queries[:100])
    print(f"tr {tr.shape}, qn {qn.shape}, max|tr|={np.abs(tr).max():.4f}, "
          f"norm check={np.linalg.norm(tr[0]):.4f}")
    print()

    run_one("AdaIVF", AdaIVF, tr, qn, nlist=256)
    run_one("NovaIVF", NovaIVF, tr, qn, nlist=256)


if __name__ == "__main__":
    main()
