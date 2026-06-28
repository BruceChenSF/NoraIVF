"""
NovaIVF Experiment Suite — Stream Simulator & Experiment Runner
"""
import sys
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import time
import os
import json
from tqdm import tqdm
from collections import defaultdict
from noraivf.datasets import generate_embodied_dataset, compute_centroids, generate_boundary_queries
from noraivf.core import (
    create_index, FrozenIVF, RebuildIVF, UpdateCentroidsIVF,
    DeDriftIVF, SPFreshIVF, NovaIVF
)


# ═══════════════════════════════════════════════════════════════
# Stream Simulator
# ═══════════════════════════════════════════════════════════════
class StreamSimulator:
    """
    Simulates a streaming vector search workload:
    interleaved insertions, deletions, and queries.
    """
    def __init__(self, train_data, query_data, ground_truth, seed=42):
        self.train = train_data
        self.queries = query_data
        self.gt = ground_truth
        self.rng = np.random.RandomState(seed)
        self.n_train = len(train_data)
        self.n_queries = len(query_data)
        self.inserted_ids = []
        self.deleted_ids = set()

    def generate_workload(self, n_batches=200, batch_size=250,
                          insert_ratio=0.45, delete_ratio=0.45, query_ratio=0.10):
        """
        Generate a streaming workload.
        Returns list of (batch_ops) where each op is (type, vector, [id/gt]).
        """
        ops_per_batch = batch_size
        total_ops = n_batches * ops_per_batch
        op_types = []
        remaining_ops = total_ops

        n_query = int(total_ops * query_ratio)
        n_delete = int(total_ops * delete_ratio)
        n_insert = total_ops - n_query - n_delete

        ops = (["insert"] * n_insert + ["delete"] * n_delete + ["query"] * n_query)
        self.rng.shuffle(ops)

        # Split into batches
        batches = []
        train_ptr = 0
        query_ptr = 0
        delete_pool = []

        for b in range(n_batches):
            batch_ops = ops[b * ops_per_batch : (b + 1) * ops_per_batch]
            batch = []
            for op_type in batch_ops:
                if op_type == "insert":
                    vec = self.train[train_ptr % self.n_train].copy()
                    batch.append(("insert", vec))
                    self.inserted_ids.append(train_ptr % self.n_train)
                    delete_pool.append(train_ptr % self.n_train)
                    train_ptr += 1
                elif op_type == "delete":
                    if len(delete_pool) > 0:
                        idx = self.rng.choice(len(delete_pool))
                        vid = delete_pool.pop(idx)
                        batch.append(("delete", vid))
                        self.deleted_ids.add(vid)
                elif op_type == "query":
                    q_idx = query_ptr % self.n_queries
                    batch.append(("query", self.queries[q_idx].copy(), self.gt[q_idx]))
                    query_ptr += 1
            batches.append(batch)
        return batches

    def generate_workload_weighted(self, n_batches=200, batch_size=250,
                                    insert_weight=0.45, delete_weight=0.45, query_weight=0.10):
        """Same as above but with explicit ratios."""
        return self.generate_workload(n_batches, batch_size, insert_weight, delete_weight, query_weight)


# ═══════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════
def recall_at_k(pred_ids, gt_ids, k=100):
    """Compute Recall@k."""
    if len(pred_ids) == 0:
        return 0.0
    gt_set = set(gt_ids[:k])
    return len(set(pred_ids[:k]) & gt_set) / k


def run_experiment(index, simulator, n_batches=200, batch_size=250,
                   insert_ratio=0.45, delete_ratio=0.45, query_ratio=0.10,
                   nprobe=8, k=100, warmup_batches=20, track_boundary=False):
    """
    Run a streaming experiment and collect metrics.
    
    If track_boundary=True, also records boundary_score for each query
    (dist_to_nearest_centroid / dist_to_second_nearest_centroid) for 
    post-hoc boundary-vs-interior analysis.
    
    Returns dict with per-batch and aggregate metrics.
    """
    batches = simulator.generate_workload(
        n_batches, batch_size, insert_ratio, delete_ratio, query_ratio)

    results = {
        "recall": [],
        "qps": [],
        "query_latencies": [],
        "batch_times": [],
        "insert_latencies": [],
        "delete_latencies": [],
    }
    if track_boundary:
        results["boundary_scores"] = []  # one per query (after warmup)

    for batch_idx, batch in enumerate(tqdm(batches, desc="Running")):
        batch_start = time.perf_counter()
        batch_queries = 0
        batch_query_time = 0.0

        for op in batch:
            if op[0] == "insert":
                vec = op[1]
                t0 = time.perf_counter()
                index.insert(vec)
                results["insert_latencies"].append(time.perf_counter() - t0)
            elif op[0] == "delete":
                vid = op[1]
                t0 = time.perf_counter()
                index.delete(vid)
                results["delete_latencies"].append(time.perf_counter() - t0)
            elif op[0] == "query":
                q_vec, gt = op[1], op[2]
                t0 = time.perf_counter()
                preds = index.search(q_vec, k=k, nprobe=nprobe)
                dt = time.perf_counter() - t0
                batch_query_time += dt
                batch_queries += 1
                if batch_idx >= warmup_batches:
                    r = recall_at_k(preds, gt, k=k)
                    results["recall"].append(r)
                    results["query_latencies"].append(dt * 1000)  # ms
                    if track_boundary:
                        # Compute boundary score: how close the query is to 
                        # the decision boundary between its nearest two centroids
                        centroids = index.centroids
                        sims = np.dot(q_vec, centroids.T)
                        top2 = -np.sort(-sims)[:2]  # top 2 cosine similarities
                        eps = 1e-8
                        d1 = np.arccos(np.clip(top2[0], -1 + eps, 1 - eps))
                        d2 = np.arccos(np.clip(top2[1], -1 + eps, 1 - eps))
                        score = d1 / (d2 + eps)  # 1.0 = exactly on boundary
                        results["boundary_scores"].append(float(score))

        batch_dt = time.perf_counter() - batch_start
        results["batch_times"].append(batch_dt)

        if batch_queries > 0:
            results["qps"].append(batch_queries / batch_query_time if batch_query_time > 0 else 0)
        else:
            results["qps"].append(0)

    # Aggregate metrics
    results["avg_recall"] = np.mean(results["recall"]) if results["recall"] else 0
    results["avg_qps"] = np.mean([q for q in results["qps"] if q > 0]) if results["qps"] else 0
    results["avg_insert_latency_us"] = np.mean(results["insert_latencies"]) * 1e6 if results["insert_latencies"] else 0
    results["avg_delete_latency_us"] = np.mean(results["delete_latencies"]) * 1e6 if results["delete_latencies"] else 0

    lats = results["query_latencies"]
    results["p50_latency"] = np.percentile(lats, 50) if lats else 0
    results["p95_latency"] = np.percentile(lats, 95) if lats else 0
    results["p99_latency"] = np.percentile(lats, 99) if lats else 0
    results["p999_latency"] = np.percentile(lats, 99.9) if lats else 0

    # Index stats
    results["cumul_maintenance_time"] = index.cumul_maintenance_time
    results["n_splits"] = index.n_splits
    results["n_merges"] = index.n_merges
    results["n_reassigns"] = getattr(index, "n_reassigns", 0)
    results["total_insert_time"] = index.total_insert_time
    results["total_delete_time"] = index.total_delete_time

    return results


def summarize_boundary_split(results, boundary_threshold=0.85, boundary_frac=0.25):
    """
    Post-hoc: split query results into boundary vs interior groups
    based on the recorded boundary scores, and compute per-group metrics.
    
    Two modes:
    - If boundary_frac > 0: use percentile split (top boundary_frac as boundary)
    - If boundary_frac <= 0: use boundary_threshold hard cutoff
    
    Returns dict with 'boundary' and 'interior' sub-dicts.
    """
    if "boundary_scores" not in results or not results["boundary_scores"]:
        return {"error": "No boundary scores recorded. Run with track_boundary=True."}
    
    scores = np.array(results["boundary_scores"])
    recalls = np.array(results["recall"])
    latencies = np.array(results["query_latencies"])
    
    cutoff = boundary_threshold
    if boundary_frac > 0:
        # Percentile-based split: top boundary_frac are boundary queries
        cutoff = np.percentile(scores, 100 * (1 - boundary_frac))
        boundary_mask = scores >= cutoff
    else:
        boundary_mask = scores >= boundary_threshold
    
    interior_mask = ~boundary_mask
    
    def stats(arr):
        if len(arr) == 0:
            return {"count": 0, "recall": 0, "p50": 0, "p99": 0}
        return {
            "count": int(len(arr["recall"])),
            "recall": float(np.mean(arr["recall"])),
            "p50": float(np.percentile(arr["latency"], 50)),
            "p99": float(np.percentile(arr["latency"], 99)),
        }
    
    b_data = {"recall": recalls[boundary_mask], "latency": latencies[boundary_mask]}
    i_data = {"recall": recalls[interior_mask], "latency": latencies[interior_mask]}
    
    n_total = len(scores)
    n_boundary = int(boundary_mask.sum())
    
    return {
        "boundary": stats(b_data),
        "interior": stats(i_data),
        "n_total": n_total,
        "n_boundary": n_boundary,
        "boundary_fraction": n_boundary / n_total if n_total > 0 else 0,
        "threshold": boundary_threshold,
        "boundary_frac": boundary_frac,
        "actual_cutoff": float(cutoff) if boundary_frac > 0 else boundary_threshold,
        "score_distribution": {
            "min": float(np.min(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(np.max(scores)),
        },
    }


# ═══════════════════════════════════════════════════════════════
# Experiment 3: QPS-Recall Sweep
# ═══════════════════════════════════════════════════════════════
def run_qps_recall_sweep(index_cls, index_kwargs, data, nlist, queries, gt,
                          nprobes=[1, 2, 4, 8, 16, 32, 64, 128], k=100,
                          n_init=10000):
    """Build index with n_init vectors, then stream-insert the rest to trigger maintenance, then sweep nprobe."""
    idx = index_cls(data[:n_init], nlist, **index_kwargs)

    # Stream-insert remaining data to trigger maintenance
    for i in range(n_init, len(data)):
        idx.insert(data[i])

    # Also do some deletions to trigger merge
    all_ids = list(idx.id_to_vec.keys())[:n_init//4]
    for vid in all_ids:
        idx.delete(vid)

    results = {"nprobe": [], "recall": [], "qps": []}
    for npb in nprobes:
        t0 = time.perf_counter()
        recalls = []
        n_queries = min(1000, len(queries))
        for i in range(n_queries):
            preds = idx.search(queries[i], k=k, nprobe=npb)
            recalls.append(recall_at_k(preds, gt[i], k=k))
        dt = time.perf_counter() - t0
        results["nprobe"].append(npb)
        results["recall"].append(np.mean(recalls))
        results["qps"].append(n_queries / dt if dt > 0 else 0)
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment 4: Microbenchmark
# ═══════════════════════════════════════════════════════════════
def run_microbenchmark(index_cls, index_kwargs, data, nlist, sizes=[10000, 20000, 50000, 100000]):
    """Measure insert/delete latency at different index sizes."""
    results = {"size": [], "insert_us": [], "delete_us": []}
    for size in sizes:
        subset = data[:min(size, len(data))]
        idx = index_cls(subset, nlist, **index_kwargs)

        # Insert latency
        insert_times = []
        n_remaining = len(data) - len(subset)
        if n_remaining <= 0:
            n_remaining = len(subset)
        for _ in range(200):
            vec = data[(len(subset) + _) % len(data)]
            t0 = time.perf_counter()
            idx.insert(vec.copy())
            insert_times.append(time.perf_counter() - t0)

        # Delete latency
        delete_times = []
        all_ids = list(idx.id_to_vec.keys())
        for _ in range(200):
            if all_ids:
                vid = all_ids.pop()
                t0 = time.perf_counter()
                idx.delete(vid)
                delete_times.append(time.perf_counter() - t0)

        results["size"].append(size)
        results["insert_us"].append(np.mean(insert_times) * 1e6)
        results["delete_us"].append(np.mean(delete_times) * 1e6)
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment 5: Long-term Stability
# ═══════════════════════════════════════════════════════════════
def run_stability(index_cls, index_kwargs, data, nlist, queries, gt,
                   n_batches=200, batch_size=250, nprobe=8, k=100):
    """Full long-term streaming stability experiment."""
    n_init = len(data) // 2
    initial = data[:n_init]
    idx = index_cls(initial, nlist, **index_kwargs)
    sim = StreamSimulator(data[n_init:], queries, gt)
    return run_experiment(idx, sim, n_batches=n_batches, batch_size=batch_size,
                          nprobe=nprobe, k=k)


# ═══════════════════════════════════════════════════════════════
# Experiment 9: Boundary Query Analysis
# ═══════════════════════════════════════════════════════════════
def run_boundary_query_analysis(index_cls, index_kwargs, data, nlist, 
                                 n_boundary=5000, nprobes=[1,2,4,8,16,32,64,128], 
                                 k=100, n_init=10000, seed=42):
    """
    Build index, generate boundary queries, and evaluate QPS-Recall
    specifically on queries near partition boundaries.
    """
    # Build index
    idx = index_cls(data[:n_init], nlist, **index_kwargs)
    
    # Stream-insert remaining data to trigger maintenance
    for i in range(n_init, len(data)):
        idx.insert(data[i])
    
    # Trigger merges via deletions
    all_ids = list(idx.id_to_vec.keys())[:n_init//4]
    for vid in all_ids:
        idx.delete(vid)
    
    # Generate boundary queries using the index's centroids
    centroids = idx.centroids.copy()
    bq, bgt = generate_boundary_queries(
        data, centroids, n_queries=n_boundary, 
        boundary_threshold=0.85, seed=seed
    )
    
    # Also evaluate on original (random) queries for comparison
    rng = np.random.RandomState(seed)
    n_clusters = len(centroids)
    d = data.shape[1]
    rand_queries = np.zeros((n_boundary, d), dtype=np.float32)
    for i in range(n_boundary):
        c = rng.choice(n_clusters)
        rand_queries[i] = (centroids[c] + rng.randn(d).astype(np.float32) * 0.25)
        norm = np.linalg.norm(rand_queries[i])
        if norm > 0:
            rand_queries[i] /= norm
    
    # Compute GT for random queries
    rand_gt = np.zeros((n_boundary, 100), dtype=np.int32)
    for i in range(0, n_boundary, 500):
        end = min(i + 500, n_boundary)
        dists = np.dot(rand_queries[i:end], data.T)
        rand_gt[i:end] = np.argsort(-dists, axis=1)[:, :100]
    
    # Evaluate on both query sets
    results = {}
    for qtype, qs, qgt in [("boundary", bq, bgt), ("random", rand_queries, rand_gt)]:
        qtype_results = {"nprobe": [], "recall": [], "qps": []}
        for npb in nprobes:
            t0 = time.perf_counter()
            recalls = []
            n_eval = min(len(qs), 2000)
            for i in range(n_eval):
                preds = idx.search(qs[i], k=k, nprobe=npb)
                recalls.append(recall_at_k(preds, qgt[i], k=k))
            dt = time.perf_counter() - t0
            qtype_results["nprobe"].append(npb)
            qtype_results["recall"].append(np.mean(recalls))
            qtype_results["qps"].append(n_eval / dt if dt > 0 else 0)
        results[qtype] = qtype_results
    
    results["n_boundary"] = n_boundary
    results["n_reassigns"] = getattr(idx, "n_reassigns", 0)
    results["cumul_maintenance_time"] = idx.cumul_maintenance_time
    
    return results


# ═══════════════════════════════════════════════════════════════
# Main: Run all experiments
# ═══════════════════════════════════════════════════════════════
def load_dataset(name):
    """Load a dataset from data/ directory. Returns (train, queries, gt)."""
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", name)
    train = np.load(os.path.join(base, "train.npy"))
    queries = np.load(os.path.join(base, "queries.npy"))
    gt = np.load(os.path.join(base, "gt.npy"))
    return train, queries, gt


DATASETS = ["ScanNetCLIP", "Ego4D"]

# Method registry: (index_class, kwargs, is_baseline_style)
METHODS = {
    "Frozen":       (FrozenIVF, {}),
    "Rebuild":      (RebuildIVF, {}),
    "UpdateCentroids": (UpdateCentroidsIVF, {}),
    "DeDrift":      (DeDriftIVF, {}),
    "SPFresh":      (SPFreshIVF, {}),
    "SPFresh-NoReassign": (SPFreshIVF, {"do_reassign": False}),
    "NovaIVF":      (NovaIVF, {"use_scdkm": True}),
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("NovaIVF Experiment Suite")
    print("=" * 60)

    all_datasets_results = {}

    for ds_name in DATASETS:
        print(f"\n{'#' * 60}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#' * 60}")

        data, queries, gt = load_dataset(ds_name)
        d = data.shape[1]
        nlist = 200
        # Use a subset for faster iteration during debugging; 
        # remove [:] for full-scale runs
        n_use = min(len(data), 50000)
        n_query_use = min(len(queries), 5000)
        data = data[:n_use]
        queries = queries[:n_query_use]
        gt = gt[:n_query_use]

        print(f"  Vectors: {data.shape}, Queries: {queries.shape}, d={d}, nlist={nlist}")

        ds_results = {}

        # ============================================================
        # Experiment 3: QPS-Recall Pareto Frontier
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 3: QPS-Recall Pareto Frontier")
        print("—" * 40)

        methods_exp3 = ["Frozen", "UpdateCentroids", "DeDrift", "SPFresh", "NovaIVF"]
        exp3 = {}
        for name in methods_exp3:
            cls, kwargs = METHODS[name]
            print(f"  {name}...", end=" ", flush=True)
            res = run_qps_recall_sweep(cls, kwargs, data, nlist, queries, gt)
            exp3[name] = res
            print(f"nprobe=8: R@100={res['recall'][4]:.4f}, QPS={res['qps'][4]:.1f}")
        ds_results["exp3_qps_recall"] = exp3

        # ============================================================
        # Experiment 5: Long-term Streaming Stability
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 5: Long-term Streaming Stability")
        print("—" * 40)

        methods_exp5 = ["Frozen", "DeDrift", "SPFresh", "NovaIVF"]
        exp5 = {}
        for name in methods_exp5:
            cls, kwargs = METHODS[name]
            print(f"  {name}...", end=" ", flush=True)
            res = run_stability(cls, kwargs, data, nlist, queries, gt,
                                n_batches=150, batch_size=200, nprobe=8)
            exp5[name] = res
            print(f"R@100={res['avg_recall']:.4f}, QPS={res['avg_qps']:.1f}, "
                  f"P99={res['p99_latency']:.2f}ms")
        ds_results["exp5_stability"] = exp5

        # ============================================================
        # Experiment 4: Microbenchmark
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 4: Insert/Delete Microbenchmark")
        print("—" * 40)

        exp4 = {}
        sizes = [10000, 20000, 30000, 40000] if n_use >= 40000 else [5000, 10000, 15000]
        for name in ["SPFresh", "NovaIVF"]:
            cls, kwargs = METHODS[name]
            print(f"  {name}...", end=" ", flush=True)
            res = run_microbenchmark(cls, kwargs, data, nlist, sizes=sizes)
            exp4[name] = res
            print(f"S={sizes[-1]}: Ins={res['insert_us'][-1]:.0f}μs, Del={res['delete_us'][-1]:.0f}μs")
        ds_results["exp4_microbench"] = exp4

        # ============================================================
        # Experiment 7: Tail Latency Distribution (from Exp5 data)
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 7: Tail Latency Distribution")
        print("—" * 40)

        exp7 = {}
        for name in methods_exp5:
            if name in exp5:
                lats = exp5[name].get("query_latencies", [])
                if lats:
                    exp7[name] = {
                        "p50": float(np.percentile(lats, 50)),
                        "p95": float(np.percentile(lats, 95)),
                        "p99": float(np.percentile(lats, 99)),
                        "p999": float(np.percentile(lats, 99.9)),
                        "latencies": lats,
                    }
                    print(f"  {name}: P50={exp7[name]['p50']:.2f}, P99={exp7[name]['p99']:.2f}, "
                          f"P99.9={exp7[name]['p999']:.2f}")
        ds_results["exp7_tail_latency"] = exp7

        # ============================================================
        # Experiment 8: Workload Robustness
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 8: Workload Robustness")
        print("—" * 40)

        workloads = {
            "Insert-heavy (90%)": (0.90, 0.05, 0.05),
            "Balanced (45/45)": (0.45, 0.45, 0.10),
            "Delete-heavy (50%)": (0.25, 0.50, 0.25),
        }
        exp8 = {}
        for wl_name, (ir, dr, qr) in workloads.items():
            exp8[wl_name] = {}
            for name in ["SPFresh", "NovaIVF"]:
                cls, kwargs = METHODS[name]
                print(f"  {wl_name} — {name}...", end=" ", flush=True)
                n_init = len(data) // 2
                idx = cls(data[:n_init], nlist, **kwargs)
                sim = StreamSimulator(data[n_init:], queries, gt)
                res = run_experiment(idx, sim, n_batches=60, batch_size=150,
                                     insert_ratio=ir, delete_ratio=dr, query_ratio=qr,
                                     nprobe=8, k=100)
                exp8[wl_name][name] = res
                print(f"R@100={res['avg_recall']:.4f}, QPS={res['avg_qps']:.1f}")
        ds_results["exp8_workload"] = exp8

        # ============================================================
        # Experiment 9: Boundary Query Analysis (Streaming + Real-time Classification)
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 9: Boundary Query Analysis (Streaming)")
        print("—" * 40)

        exp9 = {}
        for name in ["SPFresh", "NovaIVF"]:
            cls, kwargs = METHODS[name]
            print(f"  {name}...", end=" ", flush=True)
            
            # Run full streaming experiment with boundary tracking
            n_init = len(data) // 2
            idx = cls(data[:n_init], nlist, **kwargs)
            sim = StreamSimulator(data[n_init:], queries, gt)
            res = run_experiment(idx, sim, n_batches=150, batch_size=200,
                                 nprobe=8, k=100, track_boundary=True)
            
            # Post-hoc: split queries into boundary vs interior
            split = summarize_boundary_split(res, boundary_threshold=0.85)
            
            exp9[name] = {
                "raw": res,
                "split": split,
            }
            
            b = split["boundary"]
            i = split["interior"]
            print(f"Boundary({b['count']}q): R@100={b['recall']:.4f}, P99={b['p99']:.2f}ms | "
                  f"Interior({i['count']}q): R@100={i['recall']:.4f}, P99={i['p99']:.2f}ms")
        
        # Compute and report the boundary-interior gap
        print(f"\n  {'Method':<12} {'Query Type':<12} {'Recall@100':<12} {'P99(ms)':<10} {'Gap':<10}")
        print(f"  {'-'*56}")
        for name in ["SPFresh", "NovaIVF"]:
            split = exp9[name]["split"]
            for qtype in ["boundary", "interior"]:
                s = split[qtype]
                gap = exp9["NovaIVF"]["split"][qtype]["recall"] - exp9["SPFresh"]["split"][qtype]["recall"] if name == "NovaIVF" else 0
                print(f"  {name:<12} {qtype:<12} {s['recall']:<12.4f} {s['p99']:<10.2f} {gap if name == 'NovaIVF' else '':<10}")
        
        ds_results["exp9_boundary_query"] = exp9

        # ============================================================
        # Experiment 6: Cumulative Maintenance Time
        # ============================================================
        print("\n" + "—" * 40)
        print("Experiment 6: Cumulative Maintenance Time")
        print("—" * 40)

        exp6 = {}
        for name in methods_exp5:
            if name in exp5:
                exp6[name] = {
                    "cumul_maintenance_time": exp5[name].get("cumul_maintenance_time", 0),
                    "n_splits": exp5[name].get("n_splits", 0),
                    "n_merges": exp5[name].get("n_merges", 0),
                    "n_reassigns": exp5[name].get("n_reassigns", 0),
                }
                print(f"  {name}: Cumul={exp6[name]['cumul_maintenance_time']:.2f}s, "
                      f"Splits={exp6[name]['n_splits']}, Reassigns={exp6[name]['n_reassigns']}")
        ds_results["exp6_cumul_maintenance"] = exp6

        # Save per-dataset results
        all_datasets_results[ds_name] = ds_results

    # ================================================================
    # Save all results
    # ================================================================
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_datasets_results = convert(all_datasets_results)

    result_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(result_path, "w") as f:
        json.dump(all_datasets_results, f, indent=2)
    print(f"\n✅ All results saved to {result_path}")
    print(f"   File size: {os.path.getsize(result_path):,} bytes")

    return all_datasets_results


if __name__ == "__main__":
    main()
