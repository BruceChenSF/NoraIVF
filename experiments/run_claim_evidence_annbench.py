"""
Claim-evidence experiments on ann-benchmarks datasets (1M-scale protocol).

Dataset coverage (10 datasets):
  * ann-benchmarks (converted dirs under ``data/annbench/<name>/``, or raw
    HDF5 fetched with --download): sift-128-euclidean, gist-960-euclidean,
    glove-100-angular, deep-image-96-angular, mnist-784-euclidean,
    fashion-mnist-784-euclidean, audioset-128-euclidean, audioset-vggish-128.
  * embodied NPY sets (``data/<name>/``, NPY_DATASETS): CoRE50, ScanNetCLIP,
    Ego4D.

Angular datasets (glove/deep-image) store cosine ground truth on raw
vectors; the load layer L2-normalizes vectors up front so squared-L2
ordering == cosine ordering == GT ordering for all downstream metrics.
Index geometry (nlist / n_init / n_stream) is chosen by dataset size
(scale_adaptive_config), overridable per-CLI-flag.
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

from noraivf.datasets import (
    compute_l2_ground_truth,
    load_ann_benchmark_hdf5,
    load_npy_vector_dataset_l2,
)
from noraivf.core import (
    AdaIVF,
    DeDriftIVF,
    FrozenIVF,
    NovaIVF,
    RebuildIVF,
    SPFreshIVF,
    kmeans_faiss,
)
from noraivf.logging import NullLogger
from experiments.run_experiments import recall_at_k

# --------------------------------------------------------------------------- #
# logging hooks: a module-level active logger, defaulting to a no-op.
# Experiment phases emit events only when a logger is attached (set_logger).
# --------------------------------------------------------------------------- #
_LOGGER = NullLogger()


def set_logger(logger):
    """Attach an ExperimentLogger to receive hook events (or None to detach)."""
    global _LOGGER
    _LOGGER = logger if logger is not None else NullLogger()


def get_logger():
    return _LOGGER


def _hook(event_type, phase=None, **fields):
    """Emit one event on the active logger (no-op when detached)."""
    _LOGGER.event(event_type, phase=phase, **fields)


# ann-benchmarks datasets the runner knows how to load.  Each is either a
# converted directory ``data/annbench/<name>/{train,queries,gt}.npy`` (the
# canonical layout produced by scripts/prepare_annbench_datasets.py,
# scripts/prepare_audioset.py and scripts/prepare_audioset_vggish.py) or a
# raw ``data/annbench/<name>.hdf5`` file (fetched with --download).
# audioset-128-euclidean / audioset-vggish-128 are produced by the
# prepare_audioset* scripts (run separately) and share the same layout.
ANNBENCH_DATASETS = [
    "sift-128-euclidean",
    "gist-960-euclidean",
    "glove-100-angular",
    "deep-image-96-angular",
    "mnist-784-euclidean",
    "fashion-mnist-784-euclidean",
    "audioset-128-euclidean",
    "audioset-vggish-128",
]

DATASET_URLS = {
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "gist-960-euclidean": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "deep-image-96-angular": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "mnist-784-euclidean": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
}

NPY_DATASETS = {
    "core50": os.path.join("data", "core50"),
    "ScanNetCLIP": os.path.join("data", "ScanNetCLIP"),
    "Ego4D": os.path.join("data", "Ego4D"),
}


def resolve_metric(name, meta=None):
    """Resolve the distance metric for an ann-benchmarks dataset name.

    ann-benchmarks convention: filenames ending in ``-angular`` store cosine
    ground truth (GT neighbor order is by cosine similarity); everything else
    is Euclidean/L2.  When a ``meta.json`` is available it may carry an
    explicit ``metric``/``gt_basis`` annotation, which takes precedence.
    """
    if isinstance(meta, dict):
        m = meta.get("metric")
        if m in ("angular", "cosine"):
            return "angular"
        if m in ("l2", "euclidean"):
            return "euclidean"
        if meta.get("gt_basis") == "cosine":
            return "angular"
    return "angular" if str(name).endswith("-angular") else "euclidean"


def l2_normalize_rows(arr):
    """L2-normalize the rows of ``arr`` (zero rows are left as-is)."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def compute_gt_on_corpus(train, queries, k, metric, corpus_end,
                         query_chunk=512, corpus_chunk=16384):
    """Exact k-NN ground truth restricted to the first ``corpus_end`` rows.

    Used by ``--gt-scope indexed``: the index only ever covers
    ``train[:n_init + n_stream]`` (build_after_stream seeds on ``data[:n_init]``
    then streams ``data[n_init:n_init + n_stream]``), so evaluating against the
    ann-benchmarks GT -- computed over the whole train set -- under-reports
    recall.  This recomputes GT on exactly the corpus the index sees.

    Queries and corpus are both chunked (blocked matmul with a running top-k
    merged across corpus chunks) so peak memory stays bounded even at ~10M x
    96d corpora.  ``metric`` selects the scoring: "euclidean" uses squared-L2
    distance (ascending); "angular" uses inner product on the L2-normalized
    vectors stored by the load layer (descending == cosine order).

    Returns an ``(n_queries, k)`` int32 array of indices into
    ``train[:corpus_end]`` (i.e. the ids the index itself would return).
    """
    corpus_end = int(min(corpus_end, len(train)))
    k = int(min(k, corpus_end))
    n_q = len(queries)
    if k <= 0:
        return np.zeros((n_q, 0), dtype=np.int32)
    gt = np.zeros((n_q, k), dtype=np.int32)
    for i0 in range(0, n_q, query_chunk):
        i1 = min(i0 + query_chunk, n_q)
        q = np.asarray(queries[i0:i1], dtype=np.float32)
        if metric == "angular":
            top_s = np.full((i1 - i0, k), -np.inf, dtype=np.float32)
        else:
            top_s = np.full((i1 - i0, k), np.inf, dtype=np.float32)
            q_sq = np.sum(q * q, axis=1, keepdims=True)
        top_i = np.zeros((i1 - i0, k), dtype=np.int32)
        for c0 in range(0, corpus_end, corpus_chunk):
            c1 = min(c0 + corpus_chunk, corpus_end)
            corp = np.asarray(train[c0:c1], dtype=np.float32)
            prod = q @ corp.T
            n_cols = prod.shape[1]
            if n_cols > k:
                if metric == "angular":
                    scores = prod
                    part = np.argpartition(prod, -k, axis=1)[:, -k:]
                else:
                    scores = q_sq + np.sum(corp * corp, axis=1)[None, :] - 2.0 * prod
                    part = np.argpartition(scores, k - 1, axis=1)[:, :k]
                cand_s = np.take_along_axis(scores, part, axis=1)
                cand_i = part + c0
            else:
                # tiny trailing chunk (< k rows): take it whole
                if metric == "angular":
                    cand_s = prod
                else:
                    cand_s = q_sq + np.sum(corp * corp, axis=1)[None, :] - 2.0 * prod
                cand_i = np.broadcast_to(
                    np.arange(c0, c1, dtype=np.int32), (i1 - i0, c1 - c0))
            merged_s = np.concatenate([top_s, cand_s], axis=1)
            merged_i = np.concatenate([top_i, cand_i], axis=1)
            if metric == "angular":
                sel = np.argpartition(merged_s, -k, axis=1)[:, -k:]
                order = np.argsort(
                    -np.take_along_axis(merged_s, sel, axis=1), axis=1)
            else:
                sel = np.argpartition(merged_s, k - 1, axis=1)[:, :k]
                order = np.argsort(
                    np.take_along_axis(merged_s, sel, axis=1), axis=1)
            sel_sorted = np.take_along_axis(sel, order, axis=1)
            top_s = np.take_along_axis(merged_s, sel_sorted, axis=1)
            top_i = np.take_along_axis(merged_i, sel_sorted, axis=1)
        gt[i0:i1] = top_i
    return gt


def resolve_annbench_source(name, data_dir, download=False):
    """Locate the on-disk source for an ann-benchmarks dataset name.

    Returns ``(path, dataset_format, metric)`` where ``dataset_format`` is
    "npy" for a converted directory (the canonical layout, also used by
    audioset-128-euclidean) or "hdf5" for a raw HDF5 file (downloaded on
    request with --download).  Raises FileNotFoundError when neither exists.
    """
    dir_path = os.path.join(data_dir, name)
    hdf5_path = os.path.join(data_dir, f"{name}.hdf5")
    if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "train.npy")):
        meta = None
        meta_path = os.path.join(dir_path, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                meta = None
        return dir_path, "npy", resolve_metric(name, meta)
    if not os.path.exists(hdf5_path):
        if download and name in DATASET_URLS:
            download_dataset(name, data_dir)
        else:
            raise FileNotFoundError(
                f"Missing dataset {name}: no {dir_path}/train.npy and no {hdf5_path}; "
                "rerun with --download, or run scripts/prepare_audioset.py "
                "(audioset-128-euclidean) / scripts/prepare_audioset_vggish.py "
                "(audioset-vggish-128)"
            )
    return hdf5_path, "hdf5", resolve_metric(name)


def load_annbench_dataset(name, data_dir, max_train=None, max_queries=None, k=100,
                          download=False, recompute_gt=False):
    """Load one ann-benchmarks dataset, normalizing angular vectors at load.

    Preferred source is the converted directory ``data_dir/<name>/``
    (train.npy/queries.npy/gt.npy, the layout shared by
    scripts/prepare_annbench_datasets.py and scripts/prepare_audioset.py);
    falls back to the raw HDF5 ``data_dir/<name>.hdf5`` (downloaded with
    --download).  Angular sets (glove-100-angular, deep-image-96-angular)
    store cosine GT on raw vectors: vectors are L2-normalized here so that
    squared-L2 ordering == cosine ordering == GT ordering for every
    downstream search/recall computation.

    Returns ``(train, queries, gt, metric, dataset_format)``.
    """
    path, dataset_format, metric = resolve_annbench_source(
        name, data_dir, download=download
    )
    if dataset_format == "npy":
        train, queries, gt = load_npy_vector_dataset_l2(
            path, max_train=max_train, max_queries=max_queries, k=k,
            recompute_gt=recompute_gt,
        )
    else:
        train, queries, gt = load_ann_benchmark_hdf5(
            path, max_train=max_train, max_queries=max_queries, k=k
        )
    if metric == "angular":
        train = l2_normalize_rows(train)
        queries = l2_normalize_rows(queries)
        if max_train is not None and not np.all(gt < len(train)):
            gt = compute_l2_ground_truth(train, queries, k=k)
    return train, queries, gt, metric, dataset_format


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


class L2AdaIVF(L2SearchMixin, AdaIVF):
    """Ada-IVF on raw Euclidean vectors (ann-benchmarks runner).

    The core class normalizes data and uses inner-product similarity; here
    we keep the raw vectors (consistent with the other L2 baselines), use
    squared-L2 for centroid KNN/assignment, and drive the Ada-IVF temperature
    with an L2 analog of the inner-product heating rule
    (sim = 1/(1+d), so T *= (1 + eta*(1+d_c)/(1+d_nearest))).  All
    aggregate formulas (drift, reconstruction error) stay exact because the
    core maintains per-partition sums and squared-norm sums.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("normalize", False)
        super().__init__(*args, **kwargs)

    def _similarity_matrix(self, A, B):
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True)
        return -(A2 + B2.T - 2.0 * (A @ B.T))

    def search(self, query, k=100, nprobe=8):
        query = np.asarray(query, dtype=np.float32)
        sq = squared_l2_to_centroids(query, self.centroids[:self.k])
        candidates = np.argsort(sq)[: min(nprobe, self.k)]
        result = super().search(query, k=k, nprobe=nprobe)  # L2SearchMixin.search
        if len(candidates) > 0:
            d_nearest = float(sq[candidates[0]])
            ratio = (1.0 + sq[candidates]) / (1.0 + d_nearest)
            factor = np.clip(1.0 + self.eta * ratio, 1.0, 2.0)
            self.T[candidates] *= factor
            unsel = np.ones(self.k, dtype=bool)
            unsel[candidates] = False
            self.T[unsel] = np.maximum(self.T[unsel] * (1.0 - self.nu), 1.0)
        return result


METHODS = {
    "Frozen": (L2FrozenIVF, {}),
    "Rebuild": (L2RebuildIVF, {}),
    "DeDrift": (L2DeDriftIVF, {}),
    "SPFresh": (L2SPFreshIVF, {}),
    "NovaIVF": (L2NovaIVF, {"use_scdkm": True}),
    "AdaIVF": (L2AdaIVF, {}),
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
        t_phase = time.perf_counter()
        for qi in range(n_queries):
            t0 = time.perf_counter()
            pred = index.search(queries[qi], k=k, nprobe=nprobe)
            latencies.append((time.perf_counter() - t0) * 1000)
            recalls.append(recall_at_k(pred, gt[qi], k=k))
        total_s = sum(latencies) / 1000.0
        phase_s = time.perf_counter() - t_phase
        out["nprobe"].append(nprobe)
        out["recall"].append(float(np.mean(recalls)))
        out["qps"].append(float(n_queries / total_s) if total_s > 0 else 0.0)
        out["p50_ms"].append(float(np.percentile(latencies, 50)))
        out["p99_ms"].append(float(np.percentile(latencies, 99)))
        _hook("metric", phase="frontier", nprobe=nprobe, n_queries=n_queries,
              phase_s=round(phase_s, 6), qps=round(out["qps"][-1], 3),
              recall=round(out["recall"][-1], 6), p99_ms=round(out["p99_ms"][-1], 4))
    return out


def summarize_index(index, queries, gt, nprobes, k, count_npa=True):
    """Collect index + query quality metrics for one method.

    ``count_npa`` is expensive (O(N*k*d) pure-Python loop, ~10 min at 1M x 4k
    clusters); NPA is not a headline metric, so it can be skipped.
    """
    npa_s = 0.0
    if count_npa:
        t_npa = time.perf_counter()
        npa = count_npa_violations_l2(index)
        npa_s = time.perf_counter() - t_npa
    else:
        npa = {"checked": 0, "violations": 0, "violation_rate": None, "skipped": True}
    t_frontier = time.perf_counter()
    frontier = evaluate_frontier(index, queries, gt, nprobes=nprobes, k=k)
    frontier_s = time.perf_counter() - t_frontier
    summary = {
        "n_splits": int(index.n_splits),
        "n_merges": int(index.n_merges),
        "n_reassigns": int(getattr(index, "n_reassigns", 0)),
        "maintenance_time_s": float(index.cumul_maintenance_time),
        "split_clustering_time_s": float(getattr(index, "total_split_clustering_time", 0.0)),
        "split_reassign_time_s": float(getattr(index, "total_split_reassign_time", 0.0)),
        "scdkm_time_s": float(getattr(index, "total_scdkm_time", 0.0)),
        "ada_stats": getattr(index, "ada_stats", None),
        "npa": npa,
        "frontier": frontier,
    }
    _hook("metric", phase="summary", index_cls=type(index).__name__,
          n_splits=summary["n_splits"], n_merges=summary["n_merges"],
          n_reassigns=summary["n_reassigns"],
          maintenance_time_s=round(summary["maintenance_time_s"], 6),
          npa_s=round(npa_s, 6), npa_violation_rate=npa["violation_rate"],
          frontier_s=round(frontier_s, 6))
    return summary


def qps_at_recall(frontier, target_r):
    """Max QPS among operating points with recall >= target_r (ann-benchmarks style).

    Returns None when no operating point reaches ``target_r``.
    """
    best = None
    for recall, qps in zip(frontier["recall"], frontier["qps"]):
        if recall >= target_r:
            best = qps if best is None else max(best, qps)
    return best


def run_lire_l_sweep(data, queries, gt, nlist, n_init, n_stream, nprobes, k,
                     l_values, seed=42, checkpoint_every=0, count_npa=True):
    results = {}
    for lire_l in l_values:
        method_name = f"SPFresh-L{lire_l}"
        print(f"  {method_name}...", flush=True)
        _hook("phase_start", phase=f"lire_l:{lire_l}", nlist=nlist,
              n_init=n_init, n_stream=n_stream)
        try:
            idx = build_after_stream(
                L2SPFreshIVF,
                {"do_reassign": lire_l > 0, "L": lire_l},
                data,
                nlist,
                n_init,
                n_stream,
                seed=seed,
                checkpoint_every=checkpoint_every,
            )
            results[method_name] = summarize_index(idx, queries, gt, nprobes, k,
                                                   count_npa=count_npa)
            _hook("phase_end", phase=f"lire_l:{lire_l}", status="ok")
        except Exception as exc:
            _hook("error", phase=f"lire_l:{lire_l}", error_type=type(exc).__name__)
            results[method_name] = {"error": type(exc).__name__, "message": str(exc)}
            _hook("phase_end", phase=f"lire_l:{lire_l}", status="failed")
    return results


def run_maintenance_budget(results_by_method, budgets_hours,
                           target_recalls=(0.3, 0.5, 0.7, 0.9)):
    """Pick QPS@recall winners per maintenance-budget tier.

    Budgets are expressed in hours; a method is eligible in tier ``h`` when its
    ``maintenance_time_s`` <= ``h * 3600``.  For each target recall the winner
    is the eligible method with the highest QPS@recall.
    """
    out = {}
    for hours in budgets_hours:
        budget_s = float(hours) * 3600.0
        eligible = sorted(
            name for name, result in results_by_method.items()
            if result.get("maintenance_time_s", float("inf")) <= budget_s
        )
        tier = {
            "budget_hours": float(hours),
            "budget_s": budget_s,
            "eligible": eligible,
            "by_recall": {},
        }
        for target_r in target_recalls:
            qps_by_method = {}
            for name in eligible:
                qps = qps_at_recall(results_by_method[name]["frontier"], target_r)
                if qps is not None:
                    qps_by_method[name] = qps
            if qps_by_method:
                winner = max(qps_by_method, key=qps_by_method.get)
                tier["by_recall"][str(target_r)] = {
                    "method": winner,
                    "qps": qps_by_method[winner],
                    "qps_by_method": qps_by_method,
                }
            else:
                tier["by_recall"][str(target_r)] = {
                    "method": None,
                    "qps": None,
                    "qps_by_method": {},
                }
        out[str(hours)] = tier
    return out


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
        npa_rates = [r.get("npa", {}).get("violation_rate") for r in rows]
        metrics = {
            "maintenance_time_s": [r["maintenance_time_s"] for r in rows],
            "n_reassigns": [r["n_reassigns"] for r in rows],
            "recall_at_nprobe": [r["frontier"]["recall"][nprobe_index] for r in rows],
            "qps_at_nprobe": [r["frontier"]["qps"][nprobe_index] for r in rows],
            "p99_ms_at_nprobe": [r["frontier"]["p99_ms"][nprobe_index] for r in rows],
        }
        # NPA is optional (skipped with --skip-npa): only aggregate when present
        if any(v is not None for v in npa_rates):
            metrics["npa_violation_rate"] = [v for v in npa_rates if v is not None]
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
        # Clamp nlist so faiss.Kmeans never sees fewer training points than
        # clusters (e.g. nlist=8192 with a 5000-point microbench sample).
        eff_nlist = min(nlist, max(1, init.shape[0] // 2))
        idx = index_cls(init, eff_nlist, **kwargs)
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


def build_after_stream(index_cls, kwargs, data, nlist, n_init, n_stream, seed=42,
                       checkpoint_every=0):
    """Build an index on data[:n_init], then stream n_stream inserts.

    With ``checkpoint_every > 0``, emits wall-clock checkpoints (inserts done,
    elapsed, avg throughput) via the active logging hook, so the throughput
    curve of the streaming phase can be inspected for bottlenecks.
    """
    t0 = time.perf_counter()
    idx = index_cls(data[:n_init], nlist, **kwargs_with_seed(kwargs, seed))
    _hook("info", phase="build", message="index built",
          index_cls=index_cls.__name__, nlist=nlist, n_init=n_init,
          build_s=round(time.perf_counter() - t0, 6))

    n_stream = min(n_stream, len(data) - n_init)
    batch = checkpoint_every if checkpoint_every > 0 else max(n_stream, 1)
    t_stream = time.perf_counter()
    for kth, vec in enumerate(data[n_init:n_init + n_stream], start=1):
        idx.insert(vec)
        if batch and kth % batch == 0:
            elapsed = time.perf_counter() - t_stream
            _hook("checkpoint", phase="stream", index_cls=index_cls.__name__,
                  n_inserted=int(kth), elapsed_s=round(elapsed, 6),
                  insert_per_s=round(kth / elapsed, 3) if elapsed > 0 else 0.0)
    stream_s = time.perf_counter() - t_stream
    _hook("metric", phase="stream", index_cls=index_cls.__name__,
          n_stream=min(n_stream, len(data) - n_init), stream_s=round(stream_s, 6),
          avg_insert_us=round(stream_s * 1e6 / max(n_stream, 1), 3),
          n_splits=int(idx.n_splits), n_merges=int(idx.n_merges))
    return idx


def run_mixed_workload(index_cls, kwargs, data, queries, gt, nlist, n_init,
                       n_stream, k, nprobe=8, update_batch=100, query_batch=20,
                       delete_every=0, seed=42, query_sampling='uniform',
                       hot_frac=0.1, hot_weight=0.8):
    """Streaming insert/delete/query workload with optional query locality.

    ``query_sampling``:
      'uniform' (default, unchanged behaviour): the query schedule is the
        existing deterministic round-robin ``((round-1)*query_batch) % n``.
      'hotspot': a fixed hot query set of ``int(hot_frac*len(queries))``
        indices is drawn (without replacement) from the seeded RNG, then the
        whole query sequence is generated from the same RNG, drawing from the
        hot set with probability ``hot_weight`` and from all queries
        otherwise.  Using one RNG for the hot set and the full sequence keeps
        the workload reproducible.

    Per-query recall and the query index are recorded so the returned dict
    can additionally report ``recall_hot`` / ``recall_cold`` (mean recall on
    hot-set / non-hot-set queries) and ``hot_query_fraction``.
    """
    idx = index_cls(data[:n_init], nlist, **kwargs_with_seed(kwargs, seed))
    insert_latencies = []
    delete_latencies = []
    query_latencies = []
    recalls = []
    query_indices = []
    live_inserted = []
    deletes = 0
    rounds = 0
    stream_end = min(n_init + n_stream, len(data))

    # Hotspot schedule: fix the hot set first, then generate the entire query
    # sequence from the same seeded RNG (reproducible).
    hot_set = None
    seq_iter = None
    if query_sampling == 'hotspot':
        rng = np.random.RandomState(seed)
        n_hot = int(hot_frac * len(queries))
        if n_hot < 1:
            n_hot = 1
        hot_set = set(rng.choice(len(queries), size=n_hot, replace=False).tolist())
        n_rounds = (stream_end - n_init + update_batch - 1) // update_batch
        n_draws = n_rounds * min(query_batch, len(queries))
        seq = []
        for _ in range(n_draws):
            if rng.rand() < hot_weight:
                seq.append(int(rng.choice(list(hot_set))))
            else:
                seq.append(int(rng.randint(0, len(queries))))
        seq_iter = iter(seq)

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
        if seq_iter is not None:
            q_indices = [next(seq_iter) for _ in range(len(q_indices))]
        for qi in q_indices:
            t_q = time.perf_counter()
            pred = idx.search(queries[qi], k=k, nprobe=nprobe)
            query_latencies.append((time.perf_counter() - t_q) * 1000)
            recalls.append(recall_at_k(pred, exact_live_l2_neighbors(idx, queries[qi], k), k=k))
            query_indices.append(qi)

    total_query_s = sum(query_latencies) / 1000.0
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    recall_hot = 0.0
    recall_cold = 0.0
    hot_query_fraction = 0.0
    if hot_set is not None and recalls:
        hot_mask = np.array([qi in hot_set for qi in query_indices])
        rec_arr = np.asarray(recalls, dtype=np.float64)
        hot_query_fraction = float(hot_mask.mean())
        if hot_mask.any():
            recall_hot = float(rec_arr[hot_mask].mean())
        if (~hot_mask).any():
            recall_cold = float(rec_arr[~hot_mask].mean())
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
        "recall_mean": recall_mean,
        "recall_hot": recall_hot,
        "recall_cold": recall_cold,
        "hot_query_fraction": hot_query_fraction,
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


def nlist_for_size(n):
    """Map dataset size ``n`` to a cluster count (paper 1M-scale protocol).

    Buckets (all inclusive on the low end):

        n >= 800k        -> 4096   SIFT-1M, GIST-1M, GloVe, Deep1M-class
        100k <= n < 800k -> 1024   (no current dataset sits here)
        60k  <= n < 100k -> 512    MNIST / Fashion-MNIST (n = 60k)
        n < 60k          -> 128    AudioSet ~42k, CoRE50/ScanNet/Ego4D 50k
                                   (matches the paper's 30k-scale protocol)
    """
    if n >= 800_000:
        return 4096
    if n >= 100_000:
        return 1024
    if n >= 60_000:
        return 512
    return 128


def scale_adaptive_config(n, smoke=False, nlist=None, n_init=None, n_stream=None):
    """Resolve index/stream configuration for a dataset of size ``n``.

    Full-run defaults (paper 1M-scale protocol):
      * nlist    = nlist_for_size(n)      (4096 / 1024 / 512 / 128 by size)
      * n_init   = n // 2
      * n_stream = min(5000, n // 4)      (bounded streaming phase)
      * queries  = 1000                   (see --max-queries)

    Smoke runs keep their small fixed protocol (nlist=32, n_init <= 3000,
    n_stream <= 2000).  Any explicitly provided value (CLI flag) overrides
    the default.  Returns ``(nlist, n_init, n_stream)`` with n_init/n_stream
    clamped to the available data.
    """
    if smoke:
        nlist = nlist if nlist is not None else 32
        n_init = n_init if n_init is not None else min(n // 2, 3000)
        n_stream = n_stream if n_stream is not None else min(n - n_init, 2000)
    else:
        nlist = nlist if nlist is not None else nlist_for_size(n)
        n_init = n_init if n_init is not None else n // 2
        n_stream = n_stream if n_stream is not None else min(5000, n // 4)
    n_init = min(n_init, n - 1)
    n_stream = min(n_stream, n - n_init)
    return int(nlist), int(n_init), int(n_stream)


def run_dataset(name, path, smoke=False, max_train=None, max_queries=None,
                nlist=None, n_init=None, n_stream=None, dataset_format="hdf5",
                metric="euclidean", recompute_gt=False, seed=42,
                include_mixed=True, only_lire_sweep=False, l_values=None,
                skip_lire_sweep=False, methods=None, checkpoint_every=0,
                count_npa=True,
                query_sampling="uniform", hot_frac=0.1, hot_weight=0.8,
                gt_scope="full"):
    _hook("phase_start", phase=f"dataset:{name}", path=path,
          dataset_format=dataset_format, metric=metric, seed=seed)
    max_train = max_train if max_train is not None else (6000 if smoke else None)
    max_queries = max_queries if max_queries is not None else (200 if smoke else 1000)
    k = 10 if smoke else 100
    try:
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
    except Exception as exc:
        _hook("error", phase=f"dataset:{name}", message=f"load failed: {exc}")
        raise

    # Unified load-layer metric handling: angular sets carry cosine GT on raw
    # vectors, so L2-normalize here (squared-L2 order == cosine order == GT
    # order).  When a subset slice invalidated the on-disk GT, recompute it on
    # the normalized vectors (the generic loaders would have used raw ones).
    if metric == "angular":
        train = l2_normalize_rows(train)
        queries = l2_normalize_rows(queries)
        if max_train is not None and not np.all(gt < len(train)):
            gt = compute_l2_ground_truth(train, queries, k=k)

    _hook("info", phase=f"dataset:{name}", message="data loaded",
          n_train=int(len(train)), dim=int(train.shape[1]),
          n_queries=int(len(queries)), gt_k=int(gt.shape[1]), metric=metric)

    nlist, n_init, n_stream = scale_adaptive_config(
        len(train), smoke=smoke, nlist=nlist, n_init=n_init, n_stream=n_stream,
    )
    _hook("config", phase=f"dataset:{name}", n_train=int(len(train)),
          smoke=bool(smoke), metric=metric, nlist=nlist, n_init=n_init,
          n_stream=n_stream, max_queries=int(len(queries)), gt_scope=gt_scope)
    nprobes = [1, 2, 4, 8, 16] if smoke else [1, 2, 4, 8, 16, 32, 64]

    if gt_scope == "indexed":
        # Bounded-window protocol: the index only ever covers
        # train[:n_init + n_stream] (build_after_stream seeds on data[:n_init]
        # then streams data[n_init:n_init + n_stream]).  The ann-benchmarks GT
        # was computed over the whole train set, so recall against it would be
        # under-reported; recompute exact GT restricted to the indexed corpus.
        corpus_end = int(n_init + n_stream)
        t_gt = time.perf_counter()
        gt = compute_gt_on_corpus(train, queries, k, metric, corpus_end)
        gt_s = time.perf_counter() - t_gt
        _hook("info", phase=f"dataset:{name}",
              message="GT recomputed on indexed corpus", gt_scope=gt_scope,
              corpus_end=corpus_end, gt_s=round(gt_s, 6))

    results = {
        "dataset": name,
        "metric": metric,
        "format": dataset_format,
        "gt_scope": gt_scope,
        "coverage_fraction": float((n_init + n_stream) / len(train)),
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
            if methods and method_name not in methods:
                continue
            print(f"[{name}] {method_name}: build+stream", flush=True)
            _hook("phase_start", phase=f"method:{method_name}",
                  nlist=nlist, n_init=n_init, n_stream=n_stream, seed=seed)
            try:
                idx = build_after_stream(cls, kwargs, train, nlist, n_init, n_stream,
                                         seed=seed, checkpoint_every=checkpoint_every)
                results["methods"][method_name] = summarize_index(
                    idx, queries, gt, nprobes, k, count_npa=count_npa
                )
                _hook("phase_end", phase=f"method:{method_name}", status="ok")
            except Exception as exc:
                _hook("error", phase=f"method:{method_name}", error_type=type(exc).__name__)
                results["methods"][method_name] = {"error": type(exc).__name__, "message": str(exc)}
                _hook("phase_end", phase=f"method:{method_name}", status="failed")

    if skip_lire_sweep:
        results["lire_l_sweep"] = {}
    else:
        l_values = l_values or ([0, 1, 2, 5] if smoke else [0, 1, 2, 5, 10])
        print(f"[{name}] LIRE L sweep", flush=True)
        _hook("phase_start", phase=f"lire_sweep:{name}", l_values=list(l_values))
        results["lire_l_sweep"] = run_lire_l_sweep(
            train, queries, gt, nlist, n_init, n_stream, nprobes, k, l_values, seed=seed,
            checkpoint_every=checkpoint_every, count_npa=count_npa,
        )
        _hook("phase_end", phase=f"lire_sweep:{name}")

    if only_lire_sweep:
        _hook("phase_end", phase=f"dataset:{name}")
        return results

    budget_candidates = dict(results["methods"])
    budget_candidates.update(results["lire_l_sweep"])
    budgets_hours = [0.05, 0.15, 0.5] if smoke else [0.15, 0.3, 1.0, 3.0, 8.0]
    results["maintenance_budget"] = run_maintenance_budget(budget_candidates, budgets_hours)

    micro_sizes = [1000, 2000, 4000] if smoke else [5000, 10000, 20000]
    results["microbenchmark"] = {}
    for method_name in ["SPFresh", "NovaIVF"]:
        if methods and method_name not in methods:
            continue
        cls, kwargs = METHODS[method_name]
        print(f"[{name}] {method_name}: microbenchmark", flush=True)
        _hook("phase_start", phase=f"microbench:{method_name}")
        results["microbenchmark"][method_name] = run_microbenchmark(
            cls, kwargs_with_seed(kwargs, seed), train, nlist, sizes=micro_sizes, repeats=30 if smoke else 100
        )
        _hook("phase_end", phase=f"microbench:{method_name}")

    if include_mixed:
        results["mixed_workload"] = {}
        mixed_update_batch = 100 if smoke else 250
        mixed_query_batch = 10 if smoke else 25
        mixed_stream = min(n_stream, 1000 if smoke else 3000)
        for method_name, (cls, kwargs) in METHODS.items():
            if methods and method_name not in methods:
                continue
            print(f"[{name}] {method_name}: mixed workload", flush=True)
            _hook("phase_start", phase=f"mixed:{method_name}")
            try:
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
                    query_sampling=query_sampling,
                    hot_frac=hot_frac,
                    hot_weight=hot_weight,
                )
            except Exception as exc:
                _hook("error", phase=f"mixed:{method_name}", error_type=type(exc).__name__)
                results["mixed_workload"][method_name] = {"error": type(exc).__name__, "message": str(exc)}
            _hook("phase_end", phase=f"mixed:{method_name}")

    _hook("phase_end", phase=f"dataset:{name}")
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
        "gt_scope": first["gt_scope"],
        "coverage_fraction": first["coverage_fraction"],
        "k": first["k"],
        "seeds": per_seed,
        "seed_summary": aggregate_seed_results(
            per_seed,
            nprobe_index=min(3, len(first["lire_l_sweep"][next(iter(first["lire_l_sweep"]))]["frontier"]["nprobe"]) - 1),
        ),
    }


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--output", default=os.path.join("results", "claim_evidence_annbench.json"))
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="ann-benchmarks dataset names; default: all ANNBENCH_DATASETS "
                             "(unless --npy-datasets is given)")
    parser.add_argument("--npy-datasets", nargs="+", default=[])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recompute-gt", action="store_true")
    parser.add_argument("--max-train", type=int,
                        help="cap train vectors (default: full dataset for non-smoke runs)")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--nlist", type=int)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--n-stream", type=int)
    parser.add_argument("--gt-scope", choices=["full", "indexed"], default="full",
                        help="Ground-truth scope: 'full' (default) evaluates against the "
                             "ann-benchmarks GT computed over the whole train set; "
                             "'indexed' recomputes exact GT restricted to the indexed "
                             "corpus train[:n_init+n_stream] (use with bounded-window "
                             "runs where the index never sees the tail of the corpus).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument("--only-lire-sweep", action="store_true")
    parser.add_argument("--l-values", nargs="+", type=int)
    parser.add_argument("--npy-path", action="append", default=[],
                        help="Override NPY dataset path as NAME=PATH.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Restrict to these methods (Frozen, Rebuild, DeDrift, SPFresh, NovaIVF, AdaIVF).")
    parser.add_argument("--ada-tau-f", type=float, default=None,
                        help="Ada-IVF violation threshold tau_f (default 0.75).")
    parser.add_argument("--query-sampling", choices=["uniform", "hotspot"], default="uniform",
                        help="Mixed-workload query schedule: uniform (round-robin) or "
                             "hotspot (fixed hot set, hot_weight-biased draws).")
    parser.add_argument("--hot-frac", type=float, default=0.1,
                        help="Fraction of queries in the hotspot set (query_sampling=hotspot).")
    parser.add_argument("--hot-weight", type=float, default=0.8,
                        help="Probability of drawing from the hotspot set per query (hotspot).")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Emit a stream checkpoint every N inserts (0 = off).")
    parser.add_argument("--skip-lire-sweep", action="store_true",
                        help="Skip the SPFresh-L{0,1,2,5,10} LIRE L sweep phase "
                             "(5 extra full-coverage SPFresh builds).")
    parser.add_argument("--log-dir", type=str, default=os.path.join("results", "logs"),
                        help="Structured log output dir (default: results/logs)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable structured logging hooks.")
    parser.add_argument("--skip-npa", action="store_true",
                        help="Skip NPA violation counting (O(N*k*d), slow at scale; "
                             "NPA is not a headline metric).")
    args = parser.parse_args()

    npy_datasets = dict(NPY_DATASETS)
    for override in args.npy_path:
        if "=" not in override:
            raise ValueError(f"--npy-path expects NAME=PATH, got {override}")
        ds_name, ds_path = override.split("=", 1)
        npy_datasets[ds_name] = ds_path

    if args.ada_tau_f is not None:
        METHODS["AdaIVF"] = (L2AdaIVF, {"tau_f": args.ada_tau_f})


    # structured logging hook: one run per invocation, config pinned in meta.json
    hdf5_datasets = (
        args.datasets
        if args.datasets is not None
        else ([] if args.npy_datasets else list(ANNBENCH_DATASETS))
    )
    logger = None
    if not args.no_log:
        from noraivf.logging import ExperimentLogger

        logger = ExperimentLogger(
            experiment_id="claim-evidence-annbench",
            config={
                "datasets": list(hdf5_datasets) + list(args.npy_datasets),
                "smoke": args.smoke,
                "max_train": args.max_train,
                "max_queries": args.max_queries,
                "nlist": args.nlist,
                "n_init": args.n_init,
                "n_stream": args.n_stream,
                "seeds": list(args.seeds),
                "methods": args.methods,
                "checkpoint_every": args.checkpoint_every,
                "include_mixed": not args.skip_mixed,
                "query_sampling": args.query_sampling,
                "hot_frac": args.hot_frac,
                "hot_weight": args.hot_weight,
                "count_npa": not args.skip_npa,
                "gt_scope": args.gt_scope,
                "skip_lire_sweep": args.skip_lire_sweep,
                "ada_tau_f": args.ada_tau_f,
            },
            out_dir=args.log_dir,
        )
        set_logger(logger)
        for ds_name in hdf5_datasets:
            ds_path = os.path.join(args.data_dir, f"{ds_name}.hdf5")
            if os.path.exists(ds_path):
                logger.add_data_provenance(ds_path)
        for ds_name in args.npy_datasets:
            ds_path = npy_datasets[ds_name]
            if os.path.isdir(ds_path):
                for npy in sorted(Path(ds_path).glob("*.npy")):
                    logger.add_data_provenance(npy)

    all_results = {}
    for name in hdf5_datasets:
        try:
            path, dataset_format, metric = resolve_annbench_source(
                name, args.data_dir, download=args.download
            )
        except Exception as exc:
            if logger:
                logger.error(exc, phase=f"dataset:{name}", context="resolve")
            all_results[name] = {"error": type(exc).__name__, "message": str(exc)}
            continue
        try:
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
                dataset_format=dataset_format,
                metric=metric,
                recompute_gt=args.recompute_gt,
                include_mixed=not args.skip_mixed,
                only_lire_sweep=args.only_lire_sweep,
                l_values=args.l_values,
                methods=args.methods,
                checkpoint_every=args.checkpoint_every,
                count_npa=not args.skip_npa,
                query_sampling=args.query_sampling,
                hot_frac=args.hot_frac,
                hot_weight=args.hot_weight,
                gt_scope=args.gt_scope,
                skip_lire_sweep=args.skip_lire_sweep,
            )
        except Exception as exc:
            if logger:
                logger.error(exc, phase=f"dataset:{name}", context="top-level")
            all_results[name] = {"error": type(exc).__name__, "message": str(exc)}

    for name in args.npy_datasets:
        if name not in npy_datasets:
            raise KeyError(f"Unknown npy dataset {name}. Known: {sorted(npy_datasets)}")
        path = npy_datasets[name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset directory {path}")
        try:
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
                methods=args.methods,
                checkpoint_every=args.checkpoint_every,
                count_npa=not args.skip_npa,
                query_sampling=args.query_sampling,
                hot_frac=args.hot_frac,
                hot_weight=args.hot_weight,
                gt_scope=args.gt_scope,
                skip_lire_sweep=args.skip_lire_sweep,
            )
        except Exception as exc:
            if logger:
                logger.error(exc, phase=f"dataset:{name}", context="top-level")
            all_results[name] = {"error": type(exc).__name__, "message": str(exc)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if logger:
        logger.metric("run", output=args.output, n_datasets=len(all_results))
        set_logger(None)
        logger.close()
    print(f"Wrote {args.output}")


def main():
    try:
        _main()
    except SystemExit:
        raise
    except BaseException as exc:  # record fatal infra errors in the log before exiting
        _LOGGER.error(exc, phase="fatal", context="uncaught in main")
        _LOGGER.close()
        raise


if __name__ == "__main__":
    main()
