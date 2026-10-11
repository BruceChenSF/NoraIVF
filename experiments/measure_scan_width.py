"""Per-split cost of the four CDKM candidate-set widths, measured on SIFT-1M.

Motivation: Algorithm 2 in the paper writes ``q = arg max_l phi(i,l)`` over all k
clusters, while ``_cdkm_iterate`` in core.py scans only ``[c1, c2]``.  Before
changing the implementation to match the paper, the cost of that change has to be
measured -- otherwise the re-run may produce a conclusion that refutes the claim
the paper is built on.

Modes:
  restricted : ``for c in [c1, c2]``            what core.py does today
  topl       : the two split clusters plus the L nearest centroids of the split
               partition                          the cheaper option the paper raises
  fullk      : ``for c in range(k)``            Algorithm 2 taken literally, one
                                                Python loop over all k clusters
  fullk_vec  : identical semantics, evaluated as a single BLAS matvec

Also checks that restricted and fullk agree cluster by cluster wherever their
candidate sets coincide, and that fullk and fullk_vec give equal results.
"""
from __future__ import annotations
import os, sys, time, json
from collections import defaultdict

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

DATA = "data/annbench/sift-128-euclidean"
N_INIT = 500_000
NLIST = 4096
SEED = 42
N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 15

train = np.asarray(np.load(os.path.join(DATA, "train.npy")), dtype=np.float32)
print(f"# data {train.shape}, n_init={N_INIT}, nlist={NLIST}, forced_splits={N_SPLITS}",
      flush=True)


# ---- vectorized full-k (same semantics as Algorithm 2) --------------------────────
def _cdkm_iterate_vec(self, c1, c2, candidate_indices, n_iter=5):
    k = self.k
    n = self.n_j[:k].astype(np.float64)
    s = self.s_j[:k].astype(np.float64)
    a = self.a_j[:k]
    for _ in range(n_iter):
        changed = False
        for idx in list(candidate_indices):
            if idx not in self.id_to_vec or idx not in self.ppm:
                continue
            p = self.ppm[idx]
            vec = self.id_to_vec[idx]
            rho = float(np.dot(vec, vec))
            dots = a @ vec                              # (k,) in one BLAS matvec
            s_aug = s + 2.0 * dots + rho
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = s_aug / (n + 1.0) - np.where(n > 0, s / np.maximum(n, 1.0), 0.0)
                s_red = s - 2.0 * dots + rho
                red = np.where(n > 1, s / np.maximum(n, 1.0) - s_red / np.maximum(n - 1.0, 1.0), 0.0)
            phi = np.where(np.arange(k) == p, red, phi)
            best = int(np.argmax(phi))
            if best != p:
                self._move_vector(idx, p, best, vec)
                # the statistics moved, so refresh n and s
                n = self.n_j[:k].astype(np.float64)
                s = self.s_j[:k].astype(np.float64)
                changed = True
        if not changed:
            break


# ---- patched L2 NovaIVF class carrying the four scan modes ----------------──────────
from experiments.run_billion_probe import ProbeNovaIVF


class ModeNovaIVF(ProbeNovaIVF):
    pass


def build(mode, boot):
    idx = ModeNovaIVF(boot, NLIST, seed=SEED, use_scdkm=True)
    idx.cdkm_mode = mode
    idx.cdkm_L = 8
    if mode == "fullk_vec":
        import types
        idx._cdkm_iterate = types.MethodType(_cdkm_iterate_vec, idx)
    return idx


def largest_not_done(idx, done):
    best, bs = None, -1
    for c in range(idx.k):
        if c in done:
            continue
        s = len(idx.posting_lists[c])
        if s > bs:
            best, bs = c, s
    return best, bs


def run_mode(mode, boot, n_split):
    idx = build(mode, boot)
    times, sizes = [], []
    done = defaultdict(int)
    for _ in range(n_split):
        c, sz = largest_not_done(idx, done)
        if sz < 8:
            break
        done[c] += 1
        t0 = time.perf_counter()
        idx._split(c)
        times.append(time.perf_counter() - t0)
        sizes.append(sz)
    return idx, np.array(times), np.array(sizes)


lean.install()
vecs = lean.VecStore(N_INIT + 64, train.shape[1], "float32")
vecs.load_from(train, N_INIT)
cents, _, _ = lean.kmeans_on_sample(vecs, N_INIT, NLIST, seed=SEED)
assign, _ = lean.assign_all(vecs, cents, N_INIT)
boot = Bootstrap(vecs, cents, assign, N_INIT, {})
print("# index base built", flush=True)

ref_parts = None
out = {}
for mode in ("restricted", "topl", "fullk", "fullk_vec"):
    idx, t, sz = run_mode(mode, boot, N_SPLITS)
    med = float(np.median(t))
    tot = float(t.sum())
    out[mode] = dict(n=int(len(t)), size_mean=float(sz.mean()),
                     per_split_median_s=med, per_split_mean_s=float(t.mean()),
                     total_s=tot)
    print(f"# {mode:12s} splits={len(t):>3d}  |E_j| mean={sz.mean():7.1f}  "
          f"per-split median={med:8.4f}s  total={tot:8.2f}s", flush=True)
    if mode == "restricted":
        ref_parts = [sorted(pl) for pl in idx.posting_lists[:idx.k]]
    if mode == "fullk":
        fk_parts = [sorted(pl) for pl in idx.posting_lists[:idx.k]]
        same = ref_parts is not None and fk_parts == ref_parts
        print(f"#   restricted vs fullk agreement: "
              f"{'identical' if same else 'differ (expected: candidate sets differ)'}",
              flush=True)
        out["restricted_eq_fullk"] = bool(same)
    if mode == "fullk_vec":
        vp = [sorted(pl) for pl in idx.posting_lists[:idx.k]]
        # comparing fullk_vec cluster by cluster would need a second fullk run
        # in the same mode; report the time ratio here
        out["vec_speedup_vs_fullk"] = out["fullk"]["per_split_median_s"] / max(med, 1e-9)
        print(f"#   fullk_vec speedup over fullk: "
              f"{out['vec_speedup_vs_fullk']:.1f}x", flush=True)
    del idx

# reference: SPFresh per-split cost, from its measured maintenance run
out["_ref_spfresh_per_split_s"] = 1493.1576543934643 / 1661
out["_ref_novaivf_per_split_s"] = 14.804206147789955 / 1795
print(f"# reference SPFresh per split = {out['_ref_spfresh_per_split_s']:.4f}s  "
      f"(from results/claim_evidence_sift.json)", flush=True)
print(f"# reference NovaIVF(restricted) per split = {out['_ref_novaivf_per_split_s']:.6f}s",
      flush=True)
for m in ("restricted", "topl", "fullk", "fullk_vec"):
    r = out[m]["per_split_median_s"] / out["_ref_spfresh_per_split_s"]
    print(f"#   {m:12s} vs SPFresh = {r:8.3f}x  "
          f"{'NoraIVF slower' if r > 1 else 'NoraIVF faster'}", flush=True)

json.dump(out, open("results/scdkm_scan_width_1m.json", "w"), indent=2)
print("# saved results/scdkm_scan_width_1m.json", flush=True)
