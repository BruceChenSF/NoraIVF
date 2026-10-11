"""Billion-scale maintenance probe for the NoraIVF protocols.

Two modes:

  validate   reproduce the published 1M-scale protocol exactly (same data slice,
             same nlist, same seed) so the lean substrate can be checked
             against results/claim_evidence_sift.json.
  probe      build the index over a very large corpus and measure the
             per-split maintenance cost of each method, at a cluster count the
             insert-driven protocol could never reach in reasonable time.

Maintenance cost is attributed per split: ``_split`` / ``_lire_split`` are
wrapped so every event records (cluster size |E_j|, wall seconds).  That is the
quantity the reviewers' O(T|E_j|kd) argument is about.
"""
from __future__ import annotations

import argparse, json, os, resource, sys, time
from collections import defaultdict

import numpy as np

# The harness is launched from the project root (cwd); the file itself may live
# anywhere (it is copied to /home/csf/tmpbg before running), so anchor imports
# on the cwd rather than on __file__.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/csf/projects/NoraIVF")

import noraivf.core as core                     # noqa: E402
from noraivf.lean import (Bootstrap, VecStore, assign_all,  # noqa: E402
                          install, kmeans_on_sample, verify_lossless)
from noraivf.lean_methods import LeanNovaIVFVariants, CDKMVariant  # noqa: E402

import experiments.run_claim_evidence_annbench as R  # noqa: E402


# ────────────────────────────────────────────────────────────────
# L2 wrappers (mirroring the ones in the ann-benchmarks runner)
# ────────────────────────────────────────────────────────────────
class ProbeNovaIVF(R.L2SearchMixin, LeanNovaIVFVariants):
    """L2NovaIVF + selectable CDKM scan width + per-split accounting."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.total_scdkm_time = 0.0
        self.cdkm_log = []

    def insert(self, vec):
        vec = vec.astype(np.float32)
        i = self.next_id
        self.next_id += 1
        self.id_to_vec[i] = vec
        self.n += 1
        self.n_inserts += 1

        t0 = time.perf_counter()
        c = self._find_nearest_centroid(vec)
        self.posting_lists[c].append(i)
        self.ppm[i] = c
        self.positions[i] = len(self.posting_lists[c]) - 1

        self.n_j[c] += 1
        self.a_j[c] += vec
        self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))
        self.centroids[c] = self.a_j[c] / self.n_j[c]
        self.total_insert_time += time.perf_counter() - t0

        if self.n_j[c] > self.T_split:
            self._split(c)
        return i


class ProbeSPFresh(R.L2SPFreshIVF):
    pass


METHODS = {
    "Frozen":   R.L2FrozenIVF,
    "DeDrift":  R.L2DeDriftIVF,
    "SPFresh":  ProbeSPFresh,
    "NovaIVF":  ProbeNovaIVF,
}

#: which attribute drives the split, per method
SPLIT_CALL = {
    "Frozen": None,
    "DeDrift": None,
    "SPFresh": "_lire_split",
    "NovaIVF": "_split",
}


# ────────────────────────────────────────────────────────────────
def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def wrap_splits(index, method, log):
    """Record (|E_j|, seconds, cumulative) for every split event."""
    name = SPLIT_CALL.get(method)
    if name is None:
        return
    orig = getattr(index, name)

    def timed(*a, **kw):
        j = a[0] if a else kw.get("j")
        try:
            size = int(index.n_j[j]) if hasattr(index, "n_j") else len(index.posting_lists[j])
        except Exception:
            size = -1
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        log.append(dict(cluster=int(j), size_j=size,
                        seconds=time.perf_counter() - t0))
        return out

    setattr(index, name, timed)


def largest_cluster(index):
    nclusters = getattr(index, "k", len(index.posting_lists))
    best, bs = None, -1
    for c in range(nclusters):
        s = len(index.posting_lists[c])
        if s > bs:
            best, bs = c, s
    return best, bs


def force_splits(index, method, count, log, target=None):
    """Induce ``count`` splits on the largest partitions (skipping the ones
    already split by this call so each event measures a distinct partition)."""
    name = SPLIT_CALL.get(method)
    if name is None:
        return 0
    done = 0
    refilled = defaultdict(int)
    while done < count:
        if target is not None:
            cands = [(c, len(index.posting_lists[c]))
                     for c in range(getattr(index, "k", len(index.posting_lists)))
                     if len(index.posting_lists[c]) >= 8 and c not in refilled]
            if not cands:
                break
            c, sz = max(cands, key=lambda t: t[1])
        else:
            c, sz = largest_cluster(index)
        if sz < 8:
            break
        refilled[c] += 1
        getattr(index, name)(c)
        done += 1
    return done


def evaluate(index, queries, gt, nprobes, k=100, n_queries=None):
    nq = len(queries) if n_queries is None else min(n_queries, len(queries))
    out = {"nprobe": [], "recall": [], "qps": [], "p50_ms": [], "p99_ms": []}
    for nprobe in nprobes:
        lat, hit = [], []
        t0 = time.perf_counter()
        for qi in range(nq):
            q = queries[qi]
            t1 = time.perf_counter()
            got = index.search(q, k=k, nprobe=nprobe)
            lat.append((time.perf_counter() - t1) * 1e3)
            truth = set(int(x) for x in gt[qi][:k])
            hit.append(len(truth & set(map(int, got))) / k)
        el = time.perf_counter() - t0
        out["nprobe"].append(nprobe)
        out["recall"].append(float(np.mean(hit)))
        out["qps"].append(nq / el if el > 0 else 0.0)
        out["p50_ms"].append(float(np.percentile(lat, 50)))
        out["p99_ms"].append(float(np.percentile(lat, 99)))
    return out


# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train/queries/gt npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-cache", default="",
                    help="dir with centroids.npy/assign.npy (skips quantizer build)")
    ap.add_argument("--cache-out", default="", help="dir to save centroids/assign")
    ap.add_argument("--n-init", type=int, required=True)
    ap.add_argument("--n-stream", type=int, default=0)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--methods", nargs="*", default=["NovaIVF", "SPFresh"])
    ap.add_argument("--storage", default="float16", choices=["float16", "float32"])
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--nprobes", nargs="*", type=int, default=[1, 4, 16])
    ap.add_argument("--force-splits", type=int, default=0)
    ap.add_argument("--cdkm-mode", default="restricted",
                    choices=["restricted", "topl", "fullk"])
    ap.add_argument("--cdkm-L", type=int, default=5)
    ap.add_argument("--npa-sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    install()
    t_all = time.perf_counter()
    cap = a.n_init + max(a.n_stream, 0) + 4096
    d = int(np.load(os.path.join(a.data, "train.npy"), mmap_mode="r").shape[1])

    # ---- vectors --------------------------------------------------
    bad, vmax, ns = verify_lossless(os.path.join(a.data, "train.npy"),
                                    a.n_init, a.storage)
    print(f"[probe] storage={a.storage} lossless-check: {bad} mismatches "
          f"over {ns:,} sampled rows (max|x|={vmax})", flush=True)
    if bad:
        raise SystemExit("storage dtype is not lossless on this data")
    t0 = time.perf_counter()
    vecs = VecStore(cap, d, a.storage)
    vecs.load_from(np.load(os.path.join(a.data, "train.npy"), mmap_mode="r"),
                   a.n_init)
    t_load = time.perf_counter() - t0
    print(f"[probe] loaded {a.n_init:,} vectors in {t_load:.1f}s  rss={rss_gb():.1f} GB",
          flush=True)

    # ---- quantizer ------------------------------------------------
    timing = {"load_s": t_load}
    if a.index_cache:
        centroids = np.load(os.path.join(a.index_cache, "centroids.npy"))
        assignments = np.load(os.path.join(a.index_cache, "assign.npy"))
        assert len(assignments) == a.n_init, (len(assignments), a.n_init)
        assert centroids.shape[0] == a.nlist
        timing["quantizer"] = "cached"
    else:
        t0 = time.perf_counter()
        centroids, sampled, used = kmeans_on_sample(vecs, a.n_init, a.nlist,
                                                    seed=a.seed)
        t_km = time.perf_counter() - t0
        t0 = time.perf_counter()
        assignments, _ = assign_all(vecs, centroids, a.n_init)
        t_as = time.perf_counter() - t0
        timing.update(kmeans_s=t_km, assign_s=t_as, sampled=sampled, train_pts=used)
        print(f"[probe] quantizer: kmeans {t_km:.1f}s  assign {t_as:.1f}s  "
              f"rss={rss_gb():.1f} GB", flush=True)
        if a.cache_out:
            os.makedirs(a.cache_out, exist_ok=True)
            np.save(os.path.join(a.cache_out, "centroids.npy"), centroids)
            np.save(os.path.join(a.cache_out, "assign.npy"), assignments)

    boot = Bootstrap(vecs, centroids, assignments, a.n_init, timing)
    del assignments
    import gc; gc.collect()

    queries = np.asarray(np.load(os.path.join(a.data, "queries.npy")), dtype=np.float32)
    gt = np.asarray(np.load(os.path.join(a.data, "gt.npy")), dtype=np.int32)

    results = dict(config=dict(data=a.data, n_init=a.n_init, n_stream=a.n_stream,
                               nlist=a.nlist, storage=a.storage,
                               cdkm_mode=a.cdkm_mode, cdkm_L=a.cdkm_L,
                               force_splits=a.force_splits, seed=a.seed),
                   build_timing=timing, methods={})

    for method in a.methods:
        cls = METHODS[method]
        print(f"[probe] === {method} ===", flush=True)
        t0 = time.perf_counter()
        kw = {"use_scdkm": True} if method == "NovaIVF" else {}
        idx = cls(boot, a.nlist, seed=a.seed, **kw)
        t_build = time.perf_counter() - t0
        if method == "NovaIVF":
            idx.cdkm_mode = a.cdkm_mode
            idx.cdkm_L = a.cdkm_L
        print(f"[probe] {method}: index built in {t_build:.1f}s  rss={rss_gb():.1f} GB",
              flush=True)

        split_log = []
        wrap_splits(idx, method, split_log)

        t0 = time.perf_counter()
        stream_s = 0.0
        if a.n_stream:
            for vec in np.asarray(np.load(os.path.join(a.data, "train.npy"),
                                          mmap_mode="r")[a.n_init:a.n_init + a.n_stream],
                                  dtype=np.float32):
                idx.insert(vec)
            stream_s = time.perf_counter() - t0
        n_forced = 0
        if a.force_splits and SPLIT_CALL.get(method):
            t0 = time.perf_counter()
            n_forced = force_splits(idx, method, a.force_splits, split_log,
                                    target=True)
            stream_s += time.perf_counter() - t0
        print(f"[probe] {method}: stream {a.n_stream} inserts + {n_forced} forced "
              f"splits in {stream_s:.1f}s; total splits={len(split_log)}", flush=True)

        rec = dict(
            build_s=t_build,
            n_stream_inserts=int(a.n_stream),
            n_forced_splits=int(n_forced),
            n_splits=int(getattr(idx, "n_splits", len(split_log))),
            n_merges=int(getattr(idx, "n_merges", 0)),
            n_reassigns=int(getattr(idx, "n_reassigns", 0)),
            maintenance_time_s=float(getattr(idx, "total_split_time", 0.0)
                                     + getattr(idx, "total_merge_time", 0.0)),
            stream_s=stream_s,
            split_events=split_log,
            split_size_mean=float(np.mean([e["size_j"] for e in split_log])) if split_log else 0.0,
            split_seconds_median=float(np.median([e["seconds"] for e in split_log])) if split_log else 0.0,
            split_seconds_mean=float(np.mean([e["seconds"] for e in split_log])) if split_log else 0.0,
            split_seconds_total=float(np.sum([e["seconds"] for e in split_log])),
        )
        if method == "NovaIVF":
            cd = idx.cdkm_log
            rec["cdkm_mode"] = a.cdkm_mode
            rec["cdkm_L"] = a.cdkm_L
            rec["cdkm_seconds_total"] = float(np.sum([e["seconds"] for e in cd]))
            rec["cdkm_seconds_mean"] = float(np.mean([e["seconds"] for e in cd])) if cd else 0.0
            rec["cdkm_n_candidates_mean"] = float(np.mean([e["n_candidates"] for e in cd])) if cd else 0.0
            rec["cdkm_n_targets"] = int(cd[0]["n_targets"]) if cd else 0

        if not a.skip_eval:
            rec["frontier"] = evaluate(idx, queries, gt, a.nprobes,
                                       n_queries=a.max_queries)
        np.save(os.path.join(os.path.dirname(a.out) or ".",
                             f"posting_sizes_{method}.npy"),
                np.array([len(pl) for pl in idx.posting_lists[:getattr(idx, 'k', a.nlist)]],
                         dtype=np.int64))
        results["methods"][method] = rec
        print(f"[probe] {method}: split mean "
              f"{rec['split_seconds_mean']:.3f}s  median {rec['split_seconds_median']:.3f}s "
              f"|E_j| mean {rec['split_size_mean']:.0f}", flush=True)
        del idx
        gc.collect()

    results["total_s"] = time.perf_counter() - t_all
    results["peak_rss_gb"] = rss_gb()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] DONE in {results['total_s']:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
