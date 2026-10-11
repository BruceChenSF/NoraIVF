"""NovaIVF split-threshold sweep on mnist-784-euclidean (60k).

Survey finding (noraivf/core.py, NovaIVF.__init__ line ~390 / SPFreshIVF ~242):
  the split threshold is *not* hard-coded -- it is derived from the cluster
  count and scaled by a constructor knob:

      self.avg_size = self.n / max(self.k, 1)     # n = n_init at build
      self.T_split  = self.avg_size * self.split_factor   # split_factor=2.0
      self.T_merge  = self.avg_size * self.merge_factor   # merge_factor=0.25

  NovaIVF fires a split when n_j[c] > T_split (strict, hard floor: a cluster
  with < 4 vectors never splits).  T_split/T_merge are computed once at build
  and are static while k grows.  The knob is only reachable at the index
  constructor (the claim-evidence runner does not expose it as a CLI flag), so
  this driver sweeps split_factor in {1.0, 2.0, 4.0} x NovaIVF directly.

Protocol: mnist full coverage (n_init=30000, n_stream=30000), nlist=512
(paper default for 60k), 1000 queries, k=100, nprobe frontier as the 1M-scale
runs.  Writes results/nlist_sweep/mnist_threshold_sweep.json.
"""
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.run_claim_evidence_annbench import (
    L2NovaIVF,
    build_after_stream,
    load_annbench_dataset,
    summarize_index,
)

DATA_DIR = os.path.join("data", "annbench")
OUT = os.path.join("results", "nlist_sweep", "mnist_threshold_sweep.json")
DATASET = "mnist-784-euclidean"
NLIST = 512
N_INIT = 30000
N_STREAM = 30000
K = 100
N_QUERIES = 1000
NPROBES = [1, 2, 4, 8, 16, 32, 64]
SPLIT_FACTORS = [1.0, 2.0, 4.0]   # default 2.0 is the middle level
SEED = 42


def main():
    t_start = time.perf_counter()
    train, queries, gt, metric, fmt = load_annbench_dataset(
        DATASET, DATA_DIR, max_train=None, max_queries=N_QUERIES, k=K,
        download=False, recompute_gt=False,
    )
    print(f"[{DATASET}] loaded train={train.shape} queries={queries.shape} "
          f"gt={gt.shape} metric={metric} ({time.perf_counter()-t_start:.0f}s)",
          flush=True)

    avg_size = N_INIT / NLIST
    levels = {}
    for factor in SPLIT_FACTORS:
        t0 = time.perf_counter()
        print(f"[{DATASET}] NovaIVF split_factor={factor} "
              f"(T_split={avg_size*factor:.1f})...", flush=True)
        idx = build_after_stream(
            L2NovaIVF,
            {"split_factor": factor, "use_scdkm": True},
            train, NLIST, N_INIT, N_STREAM, seed=SEED,
        )
        summary = summarize_index(idx, queries, gt, NPROBES, K, count_npa=True)
        wall_s = time.perf_counter() - t0
        levels[f"split_factor_{factor}"] = {
            "method": "NovaIVF",
            "use_scdkm": True,
            "split_factor": float(factor),
            "t_split": float(avg_size * factor),
            "t_merge": float(avg_size * 0.25),
            "final_k": int(idx.k),
            "wall_s": round(wall_s, 3),
            "summary": summary,
        }
        print(f"  -> splits={summary['n_splits']} merges={summary['n_merges']} "
              f"final_k={idx.k} maint_s={summary['maintenance_time_s']:.1f} "
              f"wall={wall_s:.0f}s", flush=True)

    out = {
        "dataset": DATASET,
        "metric": metric,
        "format": fmt,
        "nlist": NLIST,
        "n_init": N_INIT,
        "n_stream": N_STREAM,
        "k": K,
        "n_queries": N_QUERIES,
        "nprobes": NPROBES,
        "seed": SEED,
        "knob": {
            "note": "Split threshold is DERIVED (not hard-coded): "
                    "T_split = (n_init/nlist) * split_factor and "
                    "T_merge = (n_init/nlist) * merge_factor; "
                    "both fixed once at build (avg_size static while k grows).",
            "constructor_knobs": ["split_factor (default 2.0)",
                                  "merge_factor (default 0.25)", "nlist", "n_init"],
            "split_trigger": "NovaIVF splits when n_j[c] > T_split (strict); "
                             "hard floor: clusters < 4 vectors never split.",
            "exposed_in_runner_cli": False,
            "default_split_factor": 2.0,
            "merge_factor": 0.25,
        },
        "levels": levels,
        "wall_total_s": round(time.perf_counter() - t_start, 3),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
