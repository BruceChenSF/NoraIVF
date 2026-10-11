"""Smoke test: AdaIVF temperature mechanism under hotspot query locality.

Ablation of the query-driven temperature updates (use_temperature True vs
False) on a mixed streaming workload with a hotspot query schedule:

  - tr = first 50000 SIFT vectors, L2-normalized; nlist = 256
  - qn = first 100 queries, L2-normalized
  - 40 rounds; each round inserts 200 vectors (tr[(r*200)%50000:...] cyclic)
    and deletes the 100 earliest live ids (scale stays bounded); then 20
    hotspot-sampled queries (hot_frac=0.1, hot_weight=0.8, seed=42) run.
  - the whole query sequence is generated ONCE with a seeded RNG and fed
    verbatim to both runs, so the two indices see identical query traffic.

GT is the vectorized brute force over the round's live vectors
(one qs @ live.T matmul for all 20 queries); no per-query Python GT loop.

调参注记 (tau_f x temperature scan, SIFT-50k hotspot, seed=42):
  - tau_f=0.75  ON: hot-cold gap -0.037 / OFF: -0.007 -- temperature amplifies
    f and causes over-maintenance (~25 reindexes) that hurts recall.
  - tau_f=1.00  ON: gap +0.062 / OFF: +0.010 -- temperature steers maintenance
    to hot partitions; the hotspot gain is ~6x the ablation (paper mechanism
    verified).  This is the recommended working point for query-bearing loads.
  - tau_f=1.25  ON: gap +0.036 / OFF: +0.010 -- gain persists, slightly less
    maintenance than at 1.00.
  - OFF at tau_f>=1.0 does ZERO maintenance and yet has the highest overall
    recall (0.945): this workload has no distribution drift, so any maintenance
    is net-harmful.  Good rebuttal material: Ada-IVF-style heuristics do
    unnecessary maintenance on stationary, drift-free loads.
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from noraivf.core import AdaIVF


def _normalize(x):
    x = np.asarray(x, dtype=np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def generate_query_sequence(n_queries, n_draws, hot_frac=0.1, hot_weight=0.8, seed=42):
    """Hotspot query schedule: fixed hot set, then hot_weight-biased draws.

    One seeded RNG produces the hot set AND the entire draw sequence, so the
    workload is reproducible and identical across runs.
    """
    rng = np.random.RandomState(seed)
    n_hot = max(1, int(hot_frac * n_queries))
    hot = rng.choice(n_queries, size=n_hot, replace=False)
    hot_set = set(hot.tolist())
    seq = []
    for _ in range(n_draws):
        if rng.rand() < hot_weight:
            seq.append(int(hot[rng.randint(0, n_hot)]))
        else:
            seq.append(int(rng.randint(0, n_queries)))
    return np.asarray(seq, dtype=np.int64), hot_set


def brute_force_gt_topk(qs, live_vecs, live_ids, k=10):
    """Vectorized GT: one matmul for all queries + argpartition top-k."""
    sims = qs @ live_vecs.T  # (n_q, n_live); cosine since vectors are unit
    topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    return [[live_ids[i] for i in row] for row in topk]


def recall_at_k(pred, gt_ids, k=10):
    if len(pred) == 0:
        return 0.0
    return len(set(pred[:k]) & set(gt_ids[:k])) / k


def run_one(name, use_temperature, tr, qn, seq, hot_set, nlist=256, k=10,
            nprobe=8, n_rounds=40, batch=200, n_delete=100, n_query=20):
    t_start = time.perf_counter()
    idx = AdaIVF(tr, nlist, use_temperature=use_temperature)
    t_build = time.perf_counter() - t_start

    insert_lat = []
    recalls = []
    qidx = []
    for r in range(n_rounds):
        start = (r * batch) % len(tr)
        for v in tr[start:start + batch]:
            t0 = time.perf_counter()
            idx.insert(v)
            insert_lat.append((time.perf_counter() - t0) * 1e6)
        for vid in sorted(idx.id_to_vec.keys())[:n_delete]:
            idx.delete(vid)

        # vectorized GT for this round's n_query queries over live vectors
        live_ids = sorted(idx.id_to_vec.keys())
        live_vecs = np.array([idx.id_to_vec[i] for i in live_ids], dtype=np.float32)
        qs = qn[seq[r * n_query:(r + 1) * n_query]]
        gt = brute_force_gt_topk(qs, live_vecs, live_ids, k=k)
        for j, qi in enumerate(seq[r * n_query:(r + 1) * n_query]):
            pred = idx.search(qn[qi], k=k, nprobe=nprobe)
            recalls.append(recall_at_k(pred, gt[j], k=k))
            qidx.append(int(qi))

    T = np.asarray(idx.T, dtype=np.float64)
    rec = np.asarray(recalls, dtype=np.float64)
    hot_mask = np.array([qi in hot_set for qi in qidx])
    rec_hot = float(rec[hot_mask].mean()) if hot_mask.any() else 0.0
    rec_cold = float(rec[~hot_mask].mean()) if (~hot_mask).any() else 0.0
    n_live = len(idx.id_to_vec)

    print(f"=== {name} (use_temperature={use_temperature}) ===")
    print(f"  build        : {t_build:.3f}s")
    print(f"  maint        : {idx.cumul_maintenance_time:.3f}s "
          f"(split {idx.total_split_time:.3f}s, merge {idx.total_merge_time:.3f}s)")
    print(f"  ada_stats    : {idx.ada_stats}")
    print(f"  T            : max {T.max():.4g}, mean {T.mean():.4g}, "
          f"p99 {np.percentile(T, 99):.4g}")
    print(f"  recall       : hot {rec_hot:.4f} (n={int(hot_mask.sum())}), "
          f"cold {rec_cold:.4f} (n={int((~hot_mask).sum())}), "
          f"hot-cold gap {rec_hot - rec_cold:+.4f} (hot frac {hot_mask.mean():.3f})")
    print(f"  insert       : p50 {np.percentile(insert_lat, 50):.1f}us, "
          f"p99 {np.percentile(insert_lat, 99):.1f}us")
    print(f"  live         : {n_live}")
    print()
    return {
        "name": name,
        "use_temperature": bool(use_temperature),
        "maint_s": float(idx.cumul_maintenance_time),
        "ada_stats": dict(idx.ada_stats),
        "T_max": float(T.max()),
        "T_mean": float(T.mean()),
        "T_p99": float(np.percentile(T, 99)),
        "recall_hot": rec_hot,
        "recall_cold": rec_cold,
        "hot_cold_gap": rec_hot - rec_cold,
        "hot_query_fraction": float(hot_mask.mean()),
        "insert_p50_us": float(np.percentile(insert_lat, 50)),
        "insert_p99_us": float(np.percentile(insert_lat, 99)),
        "n_live": n_live,
        "n_splits": idx.n_splits,
        "n_local_reindex": idx.ada_stats["n_local_reindex"],
    }


def main():
    data_dir = _ROOT / "data" / "annbench" / "sift-128-euclidean"
    train = np.load(data_dir / "train.npy")
    queries = np.load(data_dir / "queries.npy")

    tr = _normalize(train[:50000])
    qn = _normalize(queries[:100])
    print(f"tr {tr.shape}, qn {qn.shape}")

    n_rounds = 40
    n_query = 20
    seq, hot_set = generate_query_sequence(
        len(qn), n_rounds * n_query, hot_frac=0.1, hot_weight=0.8, seed=42)
    hot_draws = int(np.isin(seq, list(hot_set)).sum())
    print(f"query seq: {len(seq)} draws, hot set {sorted(hot_set)}, "
          f"hot draws {hot_draws}/{len(seq)} ({hot_draws / len(seq):.3f})")
    print()

    a = run_one("AdaIVF ON ", True, tr, qn, seq, hot_set,
                nlist=256, n_rounds=n_rounds, n_query=n_query)
    b = run_one("AdaIVF OFF", False, tr, qn, seq, hot_set,
                nlist=256, n_rounds=n_rounds, n_query=n_query)

    gap_on = a["hot_cold_gap"]
    gap_off = b["hot_cold_gap"]
    if gap_on >= gap_off:
        verdict = "temperature helps hotspot recall (ON hot-cold gap >= OFF)"
    else:
        verdict = "temperature does NOT improve hotspot recall on this smoke (ON gap < OFF)"
    print("=" * 72)
    print(f"Conclusion: T.max ON={a['T_max']:.4g} vs OFF={b['T_max']:.4g}; "
          f"hot-cold gap ON={gap_on:+.4f} vs OFF={gap_off:+.4f}; "
          f"live ON={a['n_live']} vs OFF={b['n_live']} -> {verdict}")


if __name__ == "__main__":
    main()
