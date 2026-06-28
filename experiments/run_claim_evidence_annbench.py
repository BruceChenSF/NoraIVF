"""
Claim-evidence experiments on ann-benchmarks Euclidean datasets.

This runner is intentionally separate from the CLIP/IP experiment suite because
ann-benchmarks MNIST and Fashion-MNIST use Euclidean distance on raw vectors.
"""
import sys
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import time
import urllib.request

import numpy as np

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from noraivf.core import DeDriftIVF, FrozenIVF, NovaIVF, RebuildIVF, SPFreshIVF, kmeans_faiss
from experiments.run_experiments import recall_at_k


DATASET_URLS = {
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "mnist-784-euclidean": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
}

NPY_DATASETS = {
    "core50": os.path.join("data", "core50"),
    "ScanNetCLIP": os.path.join("data", "ScanNetCLIP"),
    "Ego4D": os.path.join("data", "Ego4D"),
}


def squared_l2_to_centroids(vec, centroids):
    diff = centroids - vec.reshape(1, -1)
    return np.sum(diff * diff, axis=1)


class L2SearchMixin:
    def _find_nearest_centroid(self, vec):
        return int(np.argmin(squared_l2_to_centroids(vec.astype(np.float32), self.centroids[:self.k])))

    def search(self, query, k=100, nprobe=8):
        query = query.astype(np.float32)
        centroid_dists = squared_l2_to_centroids(query, self.centroids[:self.k])
        candidate_clusters = np.argsort(centroid_dists)[: min(nprobe, self.k)]

        heap = []
        for c in candidate_clusters:
            for idx in self.posting_lists[c]:
                if idx not in self.id_to_vec:
                    continue
                diff = query - self.id_to_vec[idx]
                dist = float(np.dot(diff, diff))
                if len(heap) < k:
                    heap.append((dist, idx))
                    heap.sort(key=lambda x: -x[0])
                elif dist < heap[0][0]:
                    heap[0] = (dist, idx)
                    heap.sort(key=lambda x: -x[0])

        heap.sort(key=lambda x: x[0])
        return [idx for _, idx in heap]


class L2SPFreshIVF(L2SearchMixin, SPFreshIVF):
    def __init__(self, *args, **kwargs):
        lire_l = kwargs.pop("L", None)
        super().__init__(*args, **kwargs)
        if lire_l is not None:
            self.L = lire_l
        self.total_split_clustering_time = 0.0
        self.total_split_reassign_time = 0.0

    def _lire_split(self, j):
        t0 = time.perf_counter()
        if self._cluster_size(j) < 4:
            return

        indices_j = list(self.posting_lists[j])
        vecs_j = np.array([self.id_to_vec[i] for i in indices_j], dtype=np.float32)

        t_cluster = time.perf_counter()
        centroids_2, assigns_2 = kmeans_faiss(vecs_j, 2, niter=10, seed=self.rng.randint(0, 2**31))
        self.total_split_clustering_time += time.perf_counter() - t_cluster

        new_c = self.k
        self.k += 1
        self.posting_lists.append([])
        self.cluster_sums.append(np.zeros(self.d, dtype=np.float32))

        new_assigns_j = []
        new_assigns_new = []
        for idx, a in zip(indices_j, assigns_2):
            if a == 0:
                new_assigns_j.append(idx)
            else:
                new_assigns_new.append(idx)
        self.posting_lists[j] = new_assigns_j
        self.posting_lists[new_c] = new_assigns_new

        if new_assigns_j:
            vecs_old = np.array([self.id_to_vec[i] for i in new_assigns_j], dtype=np.float32)
            self.centroids[j] = vecs_old.mean(axis=0)
            self.cluster_sums[j] = vecs_old.sum(axis=0)
        if new_assigns_new:
            vecs_new = np.array([self.id_to_vec[i] for i in new_assigns_new], dtype=np.float32)
            self.centroids = np.vstack([self.centroids, vecs_new.mean(axis=0, keepdims=True)])
            self.cluster_sums[new_c] = vecs_new.sum(axis=0)
        else:
            self.centroids = np.vstack([self.centroids, centroids_2[1].reshape(1, -1)])

        if self.do_reassign:
            t_reassign = time.perf_counter()
            centroid_dists = squared_l2_to_centroids(self.centroids[j], self.centroids[:self.k])
            centroid_dists[j] = np.inf
            centroid_dists[new_c] = np.inf
            top_L = np.argsort(centroid_dists)[: min(self.L, self.k - 2)]
            affected = set([j, new_c] + list(top_L))

            n_reassigned = 0
            for c in affected:
                for idx in list(self.posting_lists[c]):
                    if idx not in self.id_to_vec:
                        continue
                    vec = self.id_to_vec[idx]
                    best_c = int(np.argmin(squared_l2_to_centroids(vec, self.centroids[:self.k])))
                    if best_c != c:
                        self.posting_lists[c].remove(idx)
                        self.posting_lists[best_c].append(idx)
                        self.cluster_sums[c] -= vec
                        self.cluster_sums[best_c] += vec
                        n_reassigned += 1

            self.n_reassigns += n_reassigned
            for c in affected:
                sz = self._cluster_size(c)
                if sz > 0:
                    self.centroids[c] = self.cluster_sums[c] / sz
            self.total_split_reassign_time += time.perf_counter() - t_reassign

        self.n_splits += 1
        self.split_history.append((j, new_c, self.centroids[j].copy(), self.centroids[new_c].copy()))
        self.total_split_time += time.perf_counter() - t0

    def _merge(self, j):
        t0 = time.perf_counter()
        if self._cluster_size(j) == 0:
            return
        centroid_dists = squared_l2_to_centroids(self.centroids[j], self.centroids[:self.k])
        centroid_dists[j] = np.inf
        m = int(np.argmin(centroid_dists))
        for idx in self.posting_lists[j]:
            self.posting_lists[m].append(idx)
        self.cluster_sums[m] += self.cluster_sums[j]
        self.posting_lists[j] = []
        self.cluster_sums[j] = np.zeros(self.d, dtype=np.float32)
        if self._cluster_size(m) > 0:
            self.centroids[m] = self.cluster_sums[m] / self._cluster_size(m)
        self.n_merges += 1
        self.total_merge_time += time.perf_counter() - t0


class L2NovaIVF(L2SearchMixin, NovaIVF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_scdkm_time = 0.0

    def insert(self, vec):
        vec = vec.astype(np.float32)
        i = self.next_id
        self.next_id += 1
        self.id_to_vec[i] = vec.copy()
        self.n += 1
        self.n_inserts += 1

        t0 = time.perf_counter()
        c = self._find_nearest_centroid(vec)
        off = len(self.posting_lists[c])
        self.posting_lists[c].append(i)
        self.ppm[i] = c
        self.positions[i] = off

        self.n_j[c] += 1
        self.a_j[c] += vec
        self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))
        self.centroids[c] = self.a_j[c] / self.n_j[c]
        self.total_insert_time += time.perf_counter() - t0

        if self.n_j[c] > self.T_split:
            self._split(c)
        return i

    def _cdkm_iterate(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._cdkm_iterate(*args, **kwargs)
        self.total_scdkm_time += time.perf_counter() - t0
        return out

    def _merge(self, j):
        t0 = time.perf_counter()
        if self.n_j[j] == 0:
            return
        self.n_merges += 1
        centroid_dists = squared_l2_to_centroids(self.centroids[j], self.centroids[:self.k])
        centroid_dists[j] = np.inf
        m = int(np.argmin(centroid_dists))

        old_len_m = len(self.posting_lists[m])
        for off, idx in enumerate(list(self.posting_lists[j])):
            self.posting_lists[m].append(idx)
            self.ppm[idx] = m
            self.positions[idx] = old_len_m + off

        self.n_j[m] += self.n_j[j]
        self.a_j[m] += self.a_j[j]
        self.s_j[m] = float(np.dot(self.a_j[m], self.a_j[m]))
        self.centroids[m] = self.a_j[m] / self.n_j[m]

        self.posting_lists[j] = []
        self.n_j[j] = 0
        self.a_j[j] = np.zeros(self.d, dtype=np.float32)
        self.s_j[j] = 0.0
        self.total_merge_time += time.perf_counter() - t0


class L2FrozenIVF(L2SearchMixin, FrozenIVF):
    pass


class L2RebuildIVF(L2SearchMixin, RebuildIVF):
    pass


class L2DeDriftIVF(L2SearchMixin, DeDriftIVF):
    pass


METHODS = {
    "Frozen": (L2FrozenIVF, {}),
    "Rebuild": (L2RebuildIVF, {}),
    "DeDrift": (L2DeDriftIVF, {}),
    "SPFresh": (L2SPFreshIVF, {}),
    "NovaIVF": (L2NovaIVF, {"use_scdkm": True}),
}


def count_npa_violations_l2(index, max_points=None):
    checked = 0
    violations = 0
    n_clusters = getattr(index, "k", len(index.posting_lists))
    for c, pl in enumerate(index.posting_lists[:n_clusters]):
        for idx in pl:
            if idx not in index.id_to_vec:
                continue
            vec = index.id_to_vec[idx]
            best = int(np.argmin(squared_l2_to_centroids(vec, index.centroids[:n_clusters])))
            checked += 1
            if best != c:
                violations += 1
            if max_points is not None and checked >= max_points:
                return {
                    "checked": checked,
                    "violations": violations,
                    "violation_rate": violations / checked if checked else 0.0,
                }
    return {
        "checked": checked,
        "violations": violations,
        "violation_rate": violations / checked if checked else 0.0,
    }


def download_dataset(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = DATASET_URLS[name]
    path = os.path.join(out_dir, f"{name}.hdf5")
    if os.path.exists(path):
        return path
    print(f"Downloading {url} -> {path}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:
        out.write(response.read())
    return path


def evaluate_frontier(index, queries, gt, nprobes, k=100, n_queries=None):
    if n_queries is None:
        n_queries = len(queries)
    n_queries = min(n_queries, len(queries))
    out = {"nprobe": [], "recall": [], "qps": [], "p50_ms": [], "p99_ms": []}
    for nprobe in nprobes:
        recalls = []
        latencies = []
        for qi in range(n_queries):
            t0 = time.perf_counter()
            pred = index.search(queries[qi], k=k, nprobe=nprobe)
            latencies.append((time.perf_counter() - t0) * 1000)
            recalls.append(recall_at_k(pred, gt[qi], k=k))
        total_s = sum(latencies) / 1000.0
        out["nprobe"].append(nprobe)
        out["recall"].append(float(np.mean(recalls)))
        out["qps"].append(float(n_queries / total_s) if total_s > 0 else 0.0)
        out["p50_ms"].append(float(np.percentile(latencies, 50)))
        out["p99_ms"].append(float(np.percentile(latencies, 99)))
    return out


def summarize_index(index, queries, gt, nprobes, k):
    return {
        "n_splits": int(index.n_splits),
        "n_merges": int(index.n_merges),
        "n_reassigns": int(getattr(index, "n_reassigns", 0)),
        "maintenance_time_s": float(index.cumul_maintenance_time),
        "split_clustering_time_s": float(getattr(index, "total_split_clustering_time", 0.0)),
        "split_reassign_time_s": float(getattr(index, "total_split_reassign_time", 0.0)),
        "scdkm_time_s": float(getattr(index, "total_scdkm_time", 0.0)),
        "npa": count_npa_violations_l2(index),
        "frontier": evaluate_frontier(index, queries, gt, nprobes=nprobes, k=k),
    }


def select_best_under_budget(candidates, budget_s, nprobe_index=3):
    eligible = []
    for method, result in candidates.items():
        maint = result["maintenance_time_s"]
        if maint <= budget_s:
            frontier = result["frontier"]
            eligible.append({
                "method": method,
                "maintenance_time_s": maint,
                "recall": frontier["recall"][nprobe_index],
                "qps": frontier["qps"][nprobe_index],
                "p99_ms": frontier.get("p99_ms", [None])[nprobe_index]
                if frontier.get("p99_ms") is not None else None,
            })
    if not eligible:
        return None
    eligible.sort(key=lambda x: (x["recall"], x["qps"]), reverse=True)
    return eligible[0]


def run_lire_l_sweep(data, queries, gt, nlist, n_init, n_stream, nprobes, k,
                     l_values, seed=42):
    results = {}
    for lire_l in l_values:
        method_name = f"SPFresh-L{lire_l}"
        print(f"  {method_name}...", flush=True)
        idx = build_after_stream(
            L2SPFreshIVF,
            {"do_reassign": lire_l > 0, "L": lire_l},
            data,
            nlist,
            n_init,
            n_stream,
            seed=seed,
        )
        results[method_name] = summarize_index(idx, queries, gt, nprobes, k)
    return results


def run_maintenance_budget(results_by_method, budgets, nprobe_index=3):
    return {
        str(budget): select_best_under_budget(results_by_method, budget, nprobe_index=nprobe_index)
        for budget in budgets
    }


def metric_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_seed_results(seed_results, nprobe_index=3):
    method_names = sorted({
        method
        for result in seed_results.values()
        for method in result.get("methods", {}).keys()
    })
    summary = {
        "n_seeds": len(seed_results),
        "seeds": [int(seed) for seed in sorted(seed_results.keys(), key=lambda x: int(x))],
        "methods": {},
    }
    for method in method_names:
        rows = [
            result["methods"][method]
            for result in seed_results.values()
            if method in result.get("methods", {})
        ]
        metrics = {
            "maintenance_time_s": [r["maintenance_time_s"] for r in rows],
            "n_reassigns": [r["n_reassigns"] for r in rows],
            "npa_violation_rate": [r["npa"]["violation_rate"] for r in rows],
            "recall_at_nprobe": [r["frontier"]["recall"][nprobe_index] for r in rows],
            "qps_at_nprobe": [r["frontier"]["qps"][nprobe_index] for r in rows],
            "p99_ms_at_nprobe": [r["frontier"]["p99_ms"][nprobe_index] for r in rows],
        }
        method_summary = {}
        for metric, values in metrics.items():
            mean, std = metric_mean_std(values)
            method_summary[f"{metric}_mean"] = mean
            method_summary[f"{metric}_std"] = std
        summary["methods"][method] = method_summary
    return summary


def run_microbenchmark(index_cls, kwargs, data, nlist, sizes, repeats=100):
    out = {"size": [], "insert_us": [], "delete_us": []}
    for size in sizes:
        init = data[: min(size, len(data) // 2)]
        idx = index_cls(init, nlist, **kwargs)
        insert_times = []
        for r in range(repeats):
            vec = data[(len(init) + r) % len(data)]
            t0 = time.perf_counter()
            idx.insert(vec)
            insert_times.append((time.perf_counter() - t0) * 1e6)

        delete_times = []
        ids = list(idx.id_to_vec.keys())[-repeats:]
        for vid in ids:
            t0 = time.perf_counter()
            idx.delete(vid)
            delete_times.append((time.perf_counter() - t0) * 1e6)

        out["size"].append(int(size))
        out["insert_us"].append(float(np.mean(insert_times)))
        out["delete_us"].append(float(np.mean(delete_times)))
    return out


def kwargs_with_seed(kwargs, seed):
    out = dict(kwargs)
    out.setdefault("seed", seed)
    return out


def build_after_stream(index_cls, kwargs, data, nlist, n_init, n_stream, seed=42):
    idx = index_cls(data[:n_init], nlist, **kwargs_with_seed(kwargs, seed))
    for vec in data[n_init : n_init + n_stream]:
        idx.insert(vec)
    return idx


def run_mixed_workload(index_cls, kwargs, data, queries, gt, nlist, n_init,
                       n_stream, k, nprobe=8, update_batch=100, query_batch=20,
                       delete_every=0, seed=42):
    idx = index_cls(data[:n_init], nlist, **kwargs_with_seed(kwargs, seed))
    insert_latencies = []
    delete_latencies = []
    query_latencies = []
    recalls = []
    live_inserted = []
    deletes = 0
    rounds = 0
    stream_end = min(n_init + n_stream, len(data))

    for start in range(n_init, stream_end, update_batch):
        rounds += 1
        end = min(start + update_batch, stream_end)
        for offset, vec in enumerate(data[start:end], start=1):
            t0 = time.perf_counter()
            vid = idx.insert(vec)
            insert_latencies.append((time.perf_counter() - t0) * 1e6)
            live_inserted.append(vid)
            if delete_every and (len(insert_latencies) % delete_every == 0) and live_inserted:
                victim = live_inserted.pop(0)
                t_del = time.perf_counter()
                idx.delete(victim)
                delete_latencies.append((time.perf_counter() - t_del) * 1e6)
                deletes += 1

        q_start = ((rounds - 1) * query_batch) % len(queries)
        q_indices = [(q_start + i) % len(queries) for i in range(min(query_batch, len(queries)))]
        for qi in q_indices:
            t_q = time.perf_counter()
            pred = idx.search(queries[qi], k=k, nprobe=nprobe)
            query_latencies.append((time.perf_counter() - t_q) * 1000)
            recalls.append(recall_at_k(pred, exact_live_l2_neighbors(idx, queries[qi], k), k=k))

    total_query_s = sum(query_latencies) / 1000.0
    return {
        "rounds": int(rounds),
        "updates": int(len(insert_latencies)),
        "deletes": int(deletes),
        "insert_p50_us": float(np.percentile(insert_latencies, 50)) if insert_latencies else 0.0,
        "insert_p99_us": float(np.percentile(insert_latencies, 99)) if insert_latencies else 0.0,
        "delete_p50_us": float(np.percentile(delete_latencies, 50)) if delete_latencies else 0.0,
        "delete_p99_us": float(np.percentile(delete_latencies, 99)) if delete_latencies else 0.0,
        "query_p50_ms": float(np.percentile(query_latencies, 50)) if query_latencies else 0.0,
        "query_p99_ms": float(np.percentile(query_latencies, 99)) if query_latencies else 0.0,
        "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "qps": float(len(query_latencies) / total_query_s) if total_query_s > 0 else 0.0,
    }


def exact_live_l2_neighbors(index, query, k):
    scored = []
    query = query.astype(np.float32)
    for idx, vec in index.id_to_vec.items():
        diff = query - vec
        scored.append((float(np.dot(diff, diff)), idx))
    scored.sort(key=lambda x: x[0])
    return [idx for _, idx in scored[:k]]


def run_dataset(name, path, smoke=False, max_train=None, max_queries=None,
                nlist=None, n_init=None, n_stream=None, dataset_format="hdf5",
                recompute_gt=False, seed=42, include_mixed=True,
                only_lire_sweep=False, l_values=None):
    max_train = max_train or (6000 if smoke else 30000)
    max_queries = max_queries or (200 if smoke else 1000)
    k = 10 if smoke else 100
    if dataset_format == "hdf5":
        train, queries, gt = load_ann_benchmark_hdf5(
            path, max_train=max_train, max_queries=max_queries, k=k
        )
    elif dataset_format == "npy":
        train, queries, gt = load_npy_vector_dataset_l2(
            path,
            max_train=max_train,
            max_queries=max_queries,
            k=k,
            recompute_gt=recompute_gt,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    nlist = nlist or (32 if smoke else 128)
    n_init = n_init or min(len(train) // 2, 3000 if smoke else 15000)
    n_stream = n_stream or min(len(train) - n_init, 2000 if smoke else 15000)
    n_init = min(n_init, len(train) - 1)
    n_stream = min(n_stream, len(train) - n_init)
    nprobes = [1, 2, 4, 8, 16] if smoke else [1, 2, 4, 8, 16, 32, 64]

    results = {
        "dataset": name,
        "metric": "euclidean",
        "format": dataset_format,
        "n_train": int(len(train)),
        "n_queries": int(len(queries)),
        "nlist": int(nlist),
        "n_init": int(n_init),
        "n_stream": int(n_stream),
        "k": int(k),
        "seed": int(seed),
        "methods": {},
    }

    if not only_lire_sweep:
        for method_name, (cls, kwargs) in METHODS.items():
            print(f"[{name}] {method_name}: build+stream", flush=True)
            idx = build_after_stream(cls, kwargs, train, nlist, n_init, n_stream, seed=seed)
            results["methods"][method_name] = summarize_index(idx, queries, gt, nprobes, k)

    l_values = l_values or ([0, 1, 2, 5] if smoke else [0, 1, 2, 5, 10])
    print(f"[{name}] LIRE L sweep", flush=True)
    results["lire_l_sweep"] = run_lire_l_sweep(
        train, queries, gt, nlist, n_init, n_stream, nprobes, k, l_values, seed=seed
    )

    if only_lire_sweep:
        return results

    budget_candidates = dict(results["methods"])
    budget_candidates.update(results["lire_l_sweep"])
    budgets = [0.05, 0.15, 0.5] if smoke else [0.15, 0.3, 1.0, 3.0]
    results["maintenance_budget"] = run_maintenance_budget(
        budget_candidates,
        budgets,
        nprobe_index=min(3, len(nprobes) - 1),
    )

    micro_sizes = [1000, 2000, 4000] if smoke else [5000, 10000, 20000]
    results["microbenchmark"] = {}
    for method_name in ["SPFresh", "NovaIVF"]:
        cls, kwargs = METHODS[method_name]
        print(f"[{name}] {method_name}: microbenchmark", flush=True)
        results["microbenchmark"][method_name] = run_microbenchmark(
            cls, kwargs_with_seed(kwargs, seed), train, nlist, sizes=micro_sizes, repeats=30 if smoke else 100
        )

    if include_mixed:
        results["mixed_workload"] = {}
        mixed_update_batch = 100 if smoke else 250
        mixed_query_batch = 10 if smoke else 25
        mixed_stream = min(n_stream, 1000 if smoke else 3000)
        for method_name, (cls, kwargs) in METHODS.items():
            print(f"[{name}] {method_name}: mixed workload", flush=True)
            results["mixed_workload"][method_name] = run_mixed_workload(
                cls,
                kwargs,
                train,
                queries,
                gt,
                nlist,
                n_init,
                mixed_stream,
                k,
                nprobe=min(8, nlist),
                update_batch=mixed_update_batch,
                query_batch=mixed_query_batch,
                delete_every=10,
                seed=seed,
            )

    return results


def run_dataset_for_seeds(name, path, seeds, **kwargs):
    if len(seeds) == 1:
        return run_dataset(name, path, seed=seeds[0], **kwargs)
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = run_dataset(name, path, seed=seed, **kwargs)
    first = next(iter(per_seed.values()))
    return {
        "dataset": name,
        "metric": "euclidean",
        "format": first["format"],
        "n_train": first["n_train"],
        "n_queries": first["n_queries"],
        "nlist": first["nlist"],
        "n_init": first["n_init"],
        "n_stream": first["n_stream"],
        "k": first["k"],
        "seeds": per_seed,
        "seed_summary": aggregate_seed_results(
            per_seed,
            nprobe_index=min(3, len(first["lire_l_sweep"][next(iter(first["lire_l_sweep"]))]["frontier"]["nprobe"]) - 1),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--output", default=os.path.join("results", "claim_evidence_annbench.json"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_URLS.keys()))
    parser.add_argument("--npy-datasets", nargs="+", default=[])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recompute-gt", action="store_true")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--nlist", type=int)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--n-stream", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--only-lire-sweep", action="store_true")
    parser.add_argument("--l-values", nargs="+", type=int)
    parser.add_argument("--npy-path", action="append", default=[],
                        help="Override NPY dataset path as NAME=PATH.")
    args = parser.parse_args()

    npy_datasets = dict(NPY_DATASETS)
    for override in args.npy_path:
        if "=" not in override:
            raise ValueError(f"--npy-path expects NAME=PATH, got {override}")
        ds_name, ds_path = override.split("=", 1)
        npy_datasets[ds_name] = ds_path

    all_results = {}
    for name in args.datasets:
        if args.download:
            path = download_dataset(name, args.data_dir)
        else:
            path = os.path.join(args.data_dir, f"{name}.hdf5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset {path}; rerun with --download")
        all_results[name] = run_dataset_for_seeds(
            name,
            path,
            seeds=args.seeds,
            smoke=args.smoke,
            max_train=args.max_train,
            max_queries=args.max_queries,
            nlist=args.nlist,
            n_init=args.n_init,
            n_stream=args.n_stream,
            dataset_format="hdf5",
            recompute_gt=args.recompute_gt,
            include_mixed=not args.skip_mixed,
            only_lire_sweep=args.only_lire_sweep,
            l_values=args.l_values,
        )

    for name in args.npy_datasets:
        if name not in npy_datasets:
            raise KeyError(f"Unknown npy dataset {name}. Known: {sorted(npy_datasets)}")
        path = npy_datasets[name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset directory {path}")
        all_results[name] = run_dataset_for_seeds(
            name,
            path,
            seeds=args.seeds,
            smoke=args.smoke,
            max_train=args.max_train,
            max_queries=args.max_queries,
            nlist=args.nlist,
            n_init=args.n_init,
            n_stream=args.n_stream,
            dataset_format="npy",
            recompute_gt=args.recompute_gt,
            include_mixed=not args.skip_mixed,
            only_lire_sweep=args.only_lire_sweep,
            l_values=args.l_values,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
