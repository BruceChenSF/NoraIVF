#!/usr/bin/env python3
"""Million-scale P99 stability + mixed-workload latency experiment.

Extends the paper's P99-stability / mixed-workload protocol (previously only
run at 30k scale) to million-scale full-coverage streams.  Per method
(paper fig:stability methods: Frozen / NovaIVF / SPFresh):

  * full-coverage stream: build on ``data[:n_init]``, then ``n_batches``
    batches of ``batch_size`` inserts (each insert timed individually with
    ``time.perf_counter``).  After every batch ``delete_ratio`` (relative to
    the batch size) live vectors are sampled uniformly and deleted (each
    delete timed), then a query checkpoint of ``queries_per_checkpoint``
    queries runs at nprobe=8 (all latencies recorded, P99 reported).
  * per-checkpoint record: current index size, query P99 (ms), this-batch
    insert P99 (us) and delete P99 (us).
  * mixed phase (after the coverage stream ends): ``mixed_rounds`` rounds of
    interleaved inserts (batch_size) + deletes (delete_ratio x batch_size)
    + queries (queries_per_checkpoint).  Reports delete P99 / query P99 in
    microseconds plus recall@100, where the ground truth is the vectorized
    brute-force nearest-neighbour set over the *live* vector set (queries are
    chunked matmuls, no Python distance loops).

Method construction, data loading and metric resolution are reused verbatim
from the claim-evidence runner (``run_claim_evidence_annbench``): L2 wrapper
classes over the core implementations, L2-normalization at load time for
angular datasets, IP/squared-L2 ordering handled by the shared load layer.

Checkpoint/mixed JSON layout mirrors the 30k microbenchmark/latency results
(``results/microbenchmark_four_methods_30k.json``): per-method arrays keyed
by index size, plus a per-method ``mixed`` summary with per-round detail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.run_claim_evidence_annbench import (  # noqa: E402
    METHODS as CLAIM_METHODS,  # {Frozen: (L2FrozenIVF, {}), NovaIVF: (L2NovaIVF, {...}), ...}
    load_annbench_dataset,
    nlist_for_size,
)
from experiments.run_experiments import recall_at_k  # noqa: E402
from noraivf.logging import ExperimentLogger  # noqa: E402

# --------------------------------------------------------------------------- #
# logging hook (same event vocabulary as the claim-evidence runner)
# --------------------------------------------------------------------------- #
_LOGGER = None


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


def _hook(event_type, phase=None, **fields):
    if _LOGGER is not None:
        _LOGGER.event(event_type, phase=phase, **fields)


# --------------------------------------------------------------------------- #
# live-corpus ground truth (vectorized, chunked matmul — no Python loops)
# --------------------------------------------------------------------------- #
def exact_live_topk(queries, live_ids, vec_buf, k, query_chunk=128, corpus_chunk=16384):
    """Exact top-``k`` live-id neighbours of ``queries`` by squared-L2.

    ``live_ids`` is an int array of currently-live ids (rows of ``vec_buf``).
    For L2-normalized corpora squared-L2 ordering == inner-product ordering
    (the shared claim-runner load layer normalizes angular sets up front), so
    this GT is the metric-consistent brute force for every L2 wrapper method.
    Both queries and corpus are chunked so peak memory stays bounded even at
    ~2M-row corpora.  Returns an ``(n_queries, k)`` int64 array of *live ids*.
    """
    q = np.asarray(queries, dtype=np.float32)
    n_q = q.shape[0]
    k = int(min(k, len(live_ids)))
    gt = np.zeros((n_q, k), dtype=np.int64)
    if k <= 0:
        return gt
    q_sq = np.einsum("ij,ij->i", q, q)
    live_ids = np.asarray(live_ids)
    for i0 in range(0, n_q, query_chunk):
        i1 = min(i0 + query_chunk, n_q)
        qc = q[i0:i1]
        qc_sq = q_sq[i0:i1]
        n_rows = i1 - i0
        # running top-k over corpus chunks: (rows, k) smallest squared-L2
        best_s = np.full((n_rows, k), np.inf, dtype=np.float32)
        best_pos = np.zeros((n_rows, k), dtype=np.int64)
        for c0 in range(0, len(live_ids), corpus_chunk):
            c1 = min(c0 + corpus_chunk, len(live_ids))
            ids = live_ids[c0:c1]
            vecs = vec_buf[ids]
            v_sq = np.einsum("ij,ij->i", vecs, vecs)
            dist = qc_sq[:, None] + v_sq[None, :] - 2.0 * (qc @ vecs.T)
            n_cols = c1 - c0
            if n_cols > k:
                part = np.argpartition(dist, k - 1, axis=1)[:, :k]
                cand_s = np.take_along_axis(dist, part, axis=1)
                cand_pos = part.astype(np.int64) + c0
            else:
                cand_s = dist
                cand_pos = np.broadcast_to(
                    np.arange(c0, c1, dtype=np.int64), (n_rows, n_cols)
                )
            merged_s = np.concatenate([best_s, cand_s], axis=1)
            merged_pos = np.concatenate([best_pos, cand_pos], axis=1)
            sel = np.argpartition(merged_s, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(merged_s, sel, axis=1), axis=1)
            sel_sorted = np.take_along_axis(sel, order, axis=1)
            best_s = np.take_along_axis(merged_s, sel_sorted, axis=1)
            best_pos = np.take_along_axis(merged_pos, sel_sorted, axis=1)
        gt[i0:i1] = live_ids[best_pos]
    return gt


# --------------------------------------------------------------------------- #
# per-method run
# --------------------------------------------------------------------------- #
def _p99(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.percentile(vals, 99)) if vals.size else 0.0


def _p50(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.percentile(vals, 50)) if vals.size else 0.0


def run_method(method_name, cls, kwargs, data, queries, nlist, args, seed,
               corpus_end=None):
    """One method's full protocol; returns a JSON-safe results dict.

    ``corpus_end`` bounds the rows the index may ever see (ids are drawn from
    ``data[:corpus_end]`` in the mixed recycle phase).  For ``--gt-scope
    indexed`` this equals ``n_init + n_stream`` so the live-set brute-force GT
    is computed only over the indexed corpus, mirroring the claim runner.
    """
    d = data.shape[1]
    n_train = len(data)
    recycle_end = int(corpus_end) if corpus_end else n_train
    n_init = int(args.n_init)
    n_stream = int(min(args.n_stream, n_train - n_init))
    n_batches = int(args.n_batches)
    batch_size = int(args.batch_size)
    nprobe = int(args.nprobe)
    k = int(args.k)
    delete_ratio = float(args.delete_ratio)
    qpc = int(args.queries_per_checkpoint)
    n_rounds = int(args.mixed_rounds)

    inserts_per_round = batch_size
    deletes_per_round = max(1, int(round(delete_ratio * batch_size)))
    queries_per_round = qpc
    n_mixed_inserts = n_rounds * inserts_per_round
    n_mixed_deletes = n_rounds * deletes_per_round

    rng = np.random.RandomState(seed + 3)  # deterministic victim sampling

    print(f"  [{method_name}] build n_init={n_init} nlist={nlist} ...", flush=True)
    _hook("phase_start", phase=f"build:{method_name}", nlist=nlist, n_init=n_init)
    t0 = time.perf_counter()
    idx = cls(data[:n_init], nlist, seed=seed, **kwargs)
    build_s = time.perf_counter() - t0
    _hook("phase_end", phase=f"build:{method_name}", build_s=round(build_s, 6))

    # id == row index of vec_buf: row j holds the value inserted as id j.
    cap = n_init + n_stream + n_mixed_inserts
    vec_buf = np.empty((cap, d), dtype=np.float32)
    alive = np.zeros(cap, dtype=bool)
    vec_buf[:n_init] = data[:n_init]
    alive[:n_init] = True
    n_live = int(n_init)

    nq = len(queries)
    q_cursor = 0  # round-robin over the query set (checkpoints + mixed)

    checkpoints = {
        "batch": [],
        "index_size": [],
        "query_p99_ms": [],
        "insert_p99_us": [],
        "delete_p99_us": [],
    }

    print(f"  [{method_name}] full-coverage stream: n_stream={n_stream} "
          f"in batches of {batch_size} (n_batches={n_batches})", flush=True)
    _hook("phase_start", phase=f"stream:{method_name}", n_stream=n_stream)
    stream_s = time.perf_counter()
    n_done = 0
    n_deletes_total = 0
    batch_idx = 0
    while n_done < n_stream and batch_idx < n_batches:
        w = min(batch_size, n_stream - n_done)
        base = n_init + n_done
        ins_lat = []
        for j in range(w):
            vec = data[base + j]
            t_ins = time.perf_counter()
            vid = idx.insert(vec)
            ins_lat.append((time.perf_counter() - t_ins) * 1e6)
            vec_buf[vid] = vec
            alive[vid] = True
            n_live += 1

        # delete delete_ratio (relative to batch size) live vectors, timed
        n_del = max(1, int(round(delete_ratio * w)))
        victims = rng.choice(np.flatnonzero(alive), size=n_del, replace=False)
        del_lat = []
        for vid in victims:
            t_del = time.perf_counter()
            idx.delete(int(vid))
            del_lat.append((time.perf_counter() - t_del) * 1e6)
            alive[vid] = False
            n_live -= 1
        n_deletes_total += n_del

        # query checkpoint
        q_idx = [(q_cursor + i) % nq for i in range(qpc)]
        q_cursor = (q_cursor + qpc) % nq
        q_lat = []
        for qi in q_idx:
            t_q = time.perf_counter()
            idx.search(queries[qi], k=k, nprobe=nprobe)
            q_lat.append((time.perf_counter() - t_q) * 1e3)

        checkpoints["batch"].append(int(batch_idx + 1))
        checkpoints["index_size"].append(int(idx.n))
        checkpoints["query_p99_ms"].append(_p99(q_lat))
        checkpoints["insert_p99_us"].append(_p99(ins_lat))
        checkpoints["delete_p99_us"].append(_p99(del_lat))
        _hook("checkpoint", phase=f"stream:{method_name}", batch=int(batch_idx + 1),
              index_size=int(idx.n), query_p99_ms=round(checkpoints["query_p99_ms"][-1], 4),
              insert_p99_us=round(checkpoints["insert_p99_us"][-1], 2),
              delete_p99_us=round(checkpoints["delete_p99_us"][-1], 2))
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == min(n_batches, (n_stream + batch_size - 1) // batch_size):
            print(f"    batch {batch_idx + 1}/{min(n_batches, (n_stream + batch_size - 1) // batch_size)}"
                  f" | size={idx.n} | qP99={checkpoints['query_p99_ms'][-1]:.3f}ms "
                  f"| insP99={checkpoints['insert_p99_us'][-1]:.0f}us "
                  f"| delP99={checkpoints['delete_p99_us'][-1]:.0f}us", flush=True)
        n_done += w
        batch_idx += 1
    stream_s = time.perf_counter() - stream_s
    _hook("phase_end", phase=f"stream:{method_name}", n_inserted=int(n_done),
          n_deletes=int(n_deletes_total), stream_s=round(stream_s, 6))

    # ------------------------------------------------------------------ #
    # mixed phase: interleaved inserts + deletes + queries per round
    # ------------------------------------------------------------------ #
    print(f"  [{method_name}] mixed phase: {n_rounds} rounds "
          f"({inserts_per_round} ins / {deletes_per_round} del / {queries_per_round} q per round)",
          flush=True)
    _hook("phase_start", phase=f"mixed:{method_name}", rounds=n_rounds)
    mixed_s = time.perf_counter()
    all_q_us, all_del_us, all_ins_us = [], [], []
    per_round = []
    per_round_raw = []  # per-round raw latencies (only serialized on demand)
    rec_total, rec_count = 0.0, 0
    for r in range(n_rounds):
        # inserts: recycle the whole dataset round-robin (the coverage stream
        # already consumed the corpus; ids stay monotonic, values repeat)
        ins_us = []
        for i in range(inserts_per_round):
            row = (r * inserts_per_round + i) % recycle_end
            vec = data[row]
            t_ins = time.perf_counter()
            vid = idx.insert(vec)
            ins_us.append((time.perf_counter() - t_ins) * 1e6)
            vec_buf[vid] = vec
            alive[vid] = True
            n_live += 1

        del_us = []
        victims = rng.choice(np.flatnonzero(alive), size=deletes_per_round, replace=False)
        for vid in victims:
            t_del = time.perf_counter()
            idx.delete(int(vid))
            del_us.append((time.perf_counter() - t_del) * 1e6)
            alive[vid] = False
            n_live -= 1

        q_idx = [(q_cursor + i) % nq for i in range(queries_per_round)]
        q_cursor = (q_cursor + queries_per_round) % nq
        q_us = []
        preds = []
        for qi in q_idx:
            t_q = time.perf_counter()
            preds.append(idx.search(queries[qi], k=k, nprobe=nprobe))
            q_us.append((time.perf_counter() - t_q) * 1e6)

        # recall@100 against vectorized brute force over the live set
        live_ids = np.flatnonzero(alive)
        q_mat = np.asarray([queries[qi] for qi in q_idx], dtype=np.float32)
        gt = exact_live_topk(q_mat, live_ids, vec_buf, k=k)
        recs = [recall_at_k(p, g, k=k) for p, g in zip(preds, gt)]
        rec_total += float(np.mean(recs)) * len(recs)
        rec_count += len(recs)

        all_ins_us.extend(ins_us)
        all_del_us.extend(del_us)
        all_q_us.extend(q_us)
        if args.dump_raw_latencies:
            per_round_raw.append({
                "round": int(r + 1),
                "insert_us": [round(float(x), 3) for x in ins_us],
                "delete_us": [round(float(x), 3) for x in del_us],
                "query_us": [round(float(x), 3) for x in q_us],
            })
        per_round.append({
            "round": int(r + 1),
            "live": int(n_live),
            "inserts": int(len(ins_us)),
            "deletes": int(len(del_us)),
            "queries": int(len(q_us)),
            "insert_p99_us": _p99(ins_us),
            "delete_p99_us": _p99(del_us),
            "query_p99_us": _p99(q_us),
            "recall": float(np.mean(recs)) if recs else 0.0,
        })
        if (r + 1) % 5 == 0 or r + 1 == n_rounds:
            print(f"    round {r + 1}/{n_rounds} | live={n_live} "
                  f"| qP99={_p99(q_us):.0f}us | delP99={_p99(del_us):.0f}us "
                  f"| R@100={per_round[-1]['recall']:.4f}", flush=True)
    mixed_s = time.perf_counter() - mixed_s
    _hook("phase_end", phase=f"mixed:{method_name}", rounds=int(n_rounds),
          mixed_s=round(mixed_s, 6), recall_mean=round(rec_total / max(rec_count, 1), 6))

    result = {
        "checkpoints": checkpoints,
        "stream": {
            "n_batches": int(batch_idx),
            "n_inserts": int(n_done),
            "n_deletes": int(n_deletes_total),
            "stream_s": round(stream_s, 6),
            "avg_insert_us": round(stream_s * 1e6 / max(n_done, 1), 3),
            "n_splits": int(idx.n_splits),
            "n_merges": int(idx.n_merges),
            "n_reassigns": int(getattr(idx, "n_reassigns", 0)),
            "cumul_maintenance_time_s": float(idx.cumul_maintenance_time),
            "scdkm_time_s": float(getattr(idx, "total_scdkm_time", 0.0)),
        },
        "mixed": {
            "rounds": int(n_rounds),
            "inserts": int(len(all_ins_us)),
            "deletes": int(len(all_del_us)),
            "queries": int(len(all_q_us)),
            "live_end": int(n_live),
            "mixed_s": round(mixed_s, 6),
            "insert_p50_us": _p50(all_ins_us),
            "insert_p99_us": _p99(all_ins_us),
            "delete_p50_us": _p50(all_del_us),
            "delete_p99_us": _p99(all_del_us),
            "query_p50_us": _p50(all_q_us),
            "query_p99_us": _p99(all_q_us),
            "recall_mean": round(rec_total / max(rec_count, 1), 6),
            "per_round": per_round,
        },
        "build_s": round(build_s, 6),
    }
    if args.dump_raw_latencies:
        result["mixed"]["raw"] = {
            "insert_us": [round(float(x), 3) for x in all_ins_us],
            "delete_us": [round(float(x), 3) for x in all_del_us],
            "query_us": [round(float(x), 3) for x in all_q_us],
        }
        result["mixed"]["raw_per_round"] = per_round_raw
    return result


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
PAPER_METHODS = ["Frozen", "NovaIVF", "SPFresh"]


def run_dataset(name, args):
    data_dir = Path(args.data_dir)
    print(f"loading {name} from {data_dir} ...", flush=True)
    train, queries, gt, metric, dataset_format = load_annbench_dataset(
        name, str(data_dir), max_train=args.max_train, max_queries=None, k=100
    )
    print(f"  train={train.shape} queries={queries.shape} gt={gt.shape} "
          f"metric={metric} format={dataset_format}", flush=True)

    n_train = len(train)
    n_init = int(min(args.n_init, n_train))
    n_stream = int(min(args.n_stream, n_train - n_init))
    nlist = int(args.nlist) if args.nlist else nlist_for_size(n_train)
    if nlist > n_init // 2:
        print(f"  note: nlist={nlist} > n_init/2={n_init // 2}; faiss kmeans may "
              f"warn; keeping requested nlist.", flush=True)

    gt_scope = str(getattr(args, "gt_scope", "full"))
    # Bounded-window protocol: with gt-scope=indexed the index only ever sees
    # data[:n_init+n_stream] (stream inserts plus mixed-phase recycled rows), so
    # the exact live-set GT evaluated in the mixed phase is by construction
    # restricted to the indexed corpus (same semantics as the claim runner).
    corpus_end = int(n_init + n_stream) if gt_scope == "indexed" else int(n_train)

    config = {
        "dataset": name,
        "metric": metric,
        "format": dataset_format,
        "n_train": int(n_train),
        "n_queries": int(len(queries)),
        "n_init": n_init,
        "n_stream": n_stream,
        "nlist": nlist,
        "n_batches": int(args.n_batches),
        "batch_size": int(args.batch_size),
        "delete_ratio": float(args.delete_ratio),
        "nprobe": int(args.nprobe),
        "queries_per_checkpoint": int(args.queries_per_checkpoint),
        "mixed_rounds": int(args.mixed_rounds),
        "k": int(args.k),
        "methods": list(args.methods),
        "seed": int(args.seed),
        "gt_scope": gt_scope,
        "max_train": (int(args.max_train) if args.max_train is not None else None),
        "corpus_end": int(corpus_end),
        "dump_raw_latencies": bool(args.dump_raw_latencies),
    }
    print(f"config: {json.dumps(config)}", flush=True)

    results = {
        "config": config,
        "dataset": name,
        "n_train": int(n_train),
        "n_queries": int(len(queries)),
        "metric": metric,
        "gt_scope": gt_scope,
        "corpus_end": int(corpus_end),
        "coverage_fraction": round((n_init + n_stream) / n_train, 6),
        "methods": {},
    }
    for method_name in args.methods:
        if method_name not in CLAIM_METHODS:
            raise KeyError(f"Unknown method {method_name}. Known: {sorted(CLAIM_METHODS)}")
        cls, kwargs = CLAIM_METHODS[method_name]
        print(f"== {method_name} ==", flush=True)
        _hook("phase_start", phase=f"method:{method_name}", n_init=n_init,
              n_stream=n_stream, nlist=nlist)
        try:
            results["methods"][method_name] = run_method(
                method_name, cls, kwargs, train, queries, nlist, args, seed=args.seed,
                corpus_end=corpus_end,
            )
            _hook("phase_end", phase=f"method:{method_name}", status="ok")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _hook("error", phase=f"method:{method_name}", error_type=type(exc).__name__)
            results["methods"][method_name] = {
                "error": type(exc).__name__, "message": str(exc)
            }
            _hook("phase_end", phase=f"method:{method_name}", status="failed")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="ann-benchmarks dataset name (sift-128-euclidean, "
                             "audioset-unbal-128, mnist-784-euclidean, ...)")
    parser.add_argument("--data-dir", default=os.path.join("data", "annbench"))
    parser.add_argument("--max-train", type=int, default=None,
                        help="cap train vectors loaded (truncate the corpus; "
                             "default: full dataset). Deep-9.9M uses the full corpus.")
    parser.add_argument("--gt-scope", choices=["full", "indexed"], default="full",
                        help="Ground-truth scope: 'full' (default) keeps the whole "
                             "train corpus available to the mixed-phase recycle; "
                             "'indexed' bounds the index to train[:n_init+n_stream] "
                             "so the exact live-set GT is computed only over the "
                             "indexed corpus (same semantics as the claim runner).")
    parser.add_argument("--n-init", type=int, required=True)
    parser.add_argument("--n-stream", type=int, required=True)
    parser.add_argument("--nlist", type=int, default=None,
                        help="cluster count (default: dataset-size adaptive, "
                             "same as claim runner)")
    parser.add_argument("--n-batches", type=int, default=None,
                        help="number of stream batches (default: ceil(n-stream/batch-size))")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--delete-ratio", type=float, default=0.5,
                        help="fraction (relative to batch size) of live vectors "
                             "deleted after each stream batch / mixed round")
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--queries-per-checkpoint", type=int, default=100)
    parser.add_argument("--mixed-rounds", type=int, default=20)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="methods to run (default: Frozen NovaIVF SPFresh)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-raw-latencies", action="store_true",
                        help="store every mixed-phase insert/delete/query latency "
                             "(microseconds, float) under mixed.raw and "
                             "mixed.raw_per_round for CDF plots; JSON becomes large")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=os.path.join("results", "logs"))
    args = parser.parse_args()

    if args.n_batches is None:
        args.n_batches = -(-args.n_stream // args.batch_size)  # ceil
    if args.methods is None:
        args.methods = list(PAPER_METHODS)

    if args.output is None:
        args.output = os.path.join("results", "latency_scale", f"{args.dataset}.json")

    logger = ExperimentLogger(
        experiment_id="latency-scale",
        config={
            "dataset": args.dataset,
            "n_init": args.n_init,
            "n_stream": args.n_stream,
            "nlist": args.nlist,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "delete_ratio": args.delete_ratio,
            "nprobe": args.nprobe,
            "queries_per_checkpoint": args.queries_per_checkpoint,
            "mixed_rounds": args.mixed_rounds,
            "methods": args.methods,
            "seed": args.seed,
            "gt_scope": args.gt_scope,
            "max_train": args.max_train,
            "dump_raw_latencies": bool(args.dump_raw_latencies),
        },
        out_dir=args.log_dir,
    )
    set_logger(logger)
    src = Path(args.data_dir) / args.dataset
    for npy in sorted(Path(src).glob("*.npy")) if src.is_dir() else [src]:
        logger.add_data_provenance(npy)

    results = run_dataset(args.dataset, args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    _hook("metric", phase="run", output=str(out_path), n_methods=len(results["methods"]))
    set_logger(None)
    logger.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
