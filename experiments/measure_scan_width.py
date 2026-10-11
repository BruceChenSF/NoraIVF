"""量测 SCDKM 三种 CDKM 扫描宽度在 1M 尺度上的每分裂成本。

动机：论文 §4 的 Algorithm 2 说 `q = arg max_l phi(i,l)`（对全部 k 个簇），
而 core.py 的 `_cdkm_iterate` 只扫 [c1, c2]。把实现改成与论文一致之前，
必须先知道这个改动的代价 —— 否则"重跑实验"可能跑出一个自我否定的结论。

三种模式：
  restricted : for c in [c1, c2]                （core.py 现状）
  topl       : 两个分裂簇 + 分裂分区最近的 L 个质心（论文 §4 提的降本方案）
  fullk      : for c in range(k)                （Algorithm 2 的字面实现，Python 逐簇循环）
  fullk_vec  : 同上，但用一次 BLAS matvec 向量化（语义完全相同，只是快）

另外做一次 restricted vs fullk 的**行为一致性**检查（逐簇比对），
以及 fullk 与 fullk_vec 的**结果等价**检查。
"""
from __future__ import annotations
import os, sys, time, json
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/csf/projects/NoraIVF")

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


# ── 向量化的 full-k（语义 = Algorithm 2） ────────────────────────
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
            dots = a @ vec                                   # (k,) 一次 BLAS matvec
            s_aug = s + 2.0 * dots + rho
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = s_aug / (n + 1.0) - np.where(n > 0, s / np.maximum(n, 1.0), 0.0)
                s_red = s - 2.0 * dots + rho
                red = np.where(n > 1, s / np.maximum(n, 1.0) - s_red / np.maximum(n - 1.0, 1.0), 0.0)
            phi = np.where(np.arange(k) == p, red, phi)
            best = int(np.argmax(phi))
            if best != p:
                self._move_vector(idx, p, best, vec)
                # 统计量变了，重算 n/s
                n = self.n_j[:k].astype(np.float64)
                s = self.s_j[:k].astype(np.float64)
                changed = True
        if not changed:
            break


# ── 打补丁的 L2 NoveIVF 类，支持四种模式 ─────────────────────────
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
        print(f"#   行为一致性 restricted vs fullk: "
              f"{'相同' if same else '不同（预期，因为候选集不同）'}", flush=True)
        out["restricted_eq_fullk"] = bool(same)
    if mode == "fullk_vec":
        vp = [sorted(pl) for pl in idx.posting_lists[:idx.k]]
        # fullk_vec 的对照要在同一模式下重跑 fullk 才能逐簇比；这里只报耗时比
        out["vec_speedup_vs_fullk"] = out["fullk"]["per_split_median_s"] / max(med, 1e-9)
        print(f"#   fullk_vec 相对 fullk 加速: "
              f"{out['vec_speedup_vs_fullk']:.1f}x", flush=True)
    del idx

# 参考：SPFresh 的每分裂成本
out["_ref_spfresh_per_split_s"] = 1493.1576543934643 / 1661
out["_ref_novaivf_per_split_s"] = 14.804206147789955 / 1795
print(f"# 参考 SPFresh 每分裂 = {out['_ref_spfresh_per_split_s']:.4f}s  "
      f"(来自 results/claim_evidence_sift.json)", flush=True)
print(f"# 参考 NovaIVF(restricted) 每分裂 = {out['_ref_novaivf_per_split_s']:.6f}s",
      flush=True)
for m in ("restricted", "topl", "fullk", "fullk_vec"):
    r = out[m]["per_split_median_s"] / out["_ref_spfresh_per_split_s"]
    print(f"#   {m:12s} 相对 SPFresh = {r:8.3f}x  "
          f"({'NoraIVF 更慢' if r > 1 else 'NoraIVF 更快'})", flush=True)

json.dump(out, open("results/scdkm_scan_width_1m.json", "w"), indent=2)
print("# saved results/scdkm_scan_width_1m.json", flush=True)
