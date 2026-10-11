"""Fidelity gate for the lean substrate.

Runs the *reference* NovaIVF (noraivf.core, untouched) and the *lean* NovaIVF
(VecStore-backed, same algorithms) on identical data, identical nlist, identical
seed, and identical insert stream, then compares

  * the maintenance outcome (n_splits / n_merges / n_reassigns)
  * the resulting partition contents, cluster by cluster
  * the resulting centroids
  * the maintenance *time* (so we know the lean containers did not change the
    cost profile that the paper's numbers were measured with)

If the outcomes match and the times are close, the lean substrate is a valid
vehicle for the billion-scale measurement.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.getcwd())
# Repo root, wherever this file is run from: experiments/ sits one level below
# it, and NORAIVF_REPO_ROOT overrides the guess.
sys.path.insert(0, os.environ.get("NORAIVF_REPO_ROOT")
                or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import noraivf.core as core
import noraivf.lean as lean
from noraivf.lean import Bootstrap
import experiments.run_claim_evidence_annbench as R

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/annbench/sift-128-euclidean"
N_INIT = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
N_STREAM = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
NLIST = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
SEED = 42

train = np.asarray(np.load(os.path.join(DATA, "train.npy")), dtype=np.float32)
print(f"data {train.shape} -> use {N_INIT + N_STREAM} rows, nlist={NLIST}", flush=True)


def summarise(idx, label):
    parts = [sorted(pl) for pl in idx.posting_lists[: idx.k]]
    return dict(label=label,
                k=int(idx.k), n=int(idx.n),
                n_splits=int(idx.n_splits), n_merges=int(idx.n_merges),
                n_reassigns=int(idx.n_reassigns),
                total_split_time=float(getattr(idx, "total_split_time", 0.0)),
                total_scdkm_time=float(getattr(idx, "total_scdkm_time", 0.0)),
                parts=parts,
                centroids=np.asarray(idx.centroids[: idx.k]).copy(),
                sizes=np.array([len(p) for p in parts]))


# ── 1. reference ────────────────────────────────────────────────
t0 = time.perf_counter()
ref = R.L2NovaIVF(train[:N_INIT], NLIST, seed=SEED, use_scdkm=True)
for v in train[N_INIT:N_INIT + N_STREAM]:
    ref.insert(v)
t_ref = time.perf_counter() - t0
a = summarise(ref, "reference")
print(f"[ref ] splits={a['n_splits']} merges={a['n_merges']} reassigns={a['n_reassigns']} "
      f"split_t={a['total_split_time']:.2f}s scdkm_t={a['total_scdkm_time']:.2f}s "
      f"wall={t_ref:.1f}s", flush=True)
del ref

# ── 2. lean ─────────────────────────────────────────────────────
lean.install()
vecs = lean.VecStore(N_INIT + N_STREAM + 64, train.shape[1], "float32")
vecs.load_from(train, N_INIT)
cents, sampled, used = lean.kmeans_on_sample(vecs, N_INIT, NLIST, seed=SEED)
assign, _ = lean.assign_all(vecs, cents, N_INIT)
boot = Bootstrap(vecs, cents, assign, N_INIT, {})

t0 = time.perf_counter()
from experiments.run_billion_probe import ProbeNovaIVF  # noqa: E402

idx = ProbeNovaIVF(boot, NLIST, seed=SEED, use_scdkm=True)
idx.cdkm_mode = "restricted"
for v in train[N_INIT:N_INIT + N_STREAM]:
    idx.insert(v)
t_lean = time.perf_counter() - t0
b = summarise(idx, "lean")
print(f"[lean] splits={b['n_splits']} merges={b['n_merges']} reassigns={b['n_reassigns']} "
      f"split_t={b['total_split_time']:.2f}s scdkm_t={sum(e['seconds'] for e in idx.cdkm_log):.2f}s "
      f"wall={t_lean:.1f}s", flush=True)

# ── 3. compare ──────────────────────────────────────────────────
print("=" * 70)
ok = True
for key in ("k", "n", "n_splits", "n_merges", "n_reassigns"):
    same = a[key] == b[key]
    ok &= same
    print(f"  {key:14s} ref={a[key]:>10} lean={b[key]:>10}  {'OK' if same else 'MISMATCH'}")

if a["k"] == b["k"]:
    diff_parts = sum(1 for x, y in zip(a["parts"], b["parts"]) if x != y)
    print(f"  partitions differing: {diff_parts}/{a['k']}")
    ok &= diff_parts == 0
    cd = float(np.abs(a["centroids"] - b["centroids"]).max())
    print(f"  max |centroid diff|: {cd:.3e}")
    ok &= cd == 0.0
    sd = np.abs(a["sizes"] - b["sizes"]).max()
    print(f"  max cluster-size diff: {sd}")

print(f"  split time  ref={a['total_split_time']:.2f}s lean={b['total_split_time']:.2f}s "
      f"ratio={b['total_split_time']/max(a['total_split_time'],1e-9):.3f}")
print(f"  wall        ref={t_ref:.1f}s lean={t_lean:.1f}s "
      f"ratio={t_lean/max(t_ref,1e-9):.3f}")
print("=" * 70)
print("FIDELITY", "PASS" if ok else "FAIL", flush=True)
json.dump({"reference": {k: v for k, v in a.items() if k not in ("parts", "centroids")},
           "lean": {k: v for k, v in b.items() if k not in ("parts", "centroids")},
           "wall_ref_s": t_ref, "wall_lean_s": t_lean,
           "fidelity": "PASS" if ok else "FAIL"},
          open("results/lean_fidelity.json", "w"), indent=2,
          default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
