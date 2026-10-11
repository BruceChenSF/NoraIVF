"""Lean-scaled index classes + the top-L / full-k CDKM variants.

Two things live here:

1. ``LeanNovaIVF`` -- NovaIVF with an array-backed PostingPointerMap.  The
   reference ``NovaIVF.__init__`` builds ``self.ppm`` and ``self.positions`` as
   plain Python dicts with one entry per vector (~200 GB at 1e9), so it cannot
   be used at billion scale even with the lean base init.  The body below is the
   reference body with ``{}`` replaced by ``IntMap``; nothing else differs.

2. ``CDKMVariant`` -- a drop-in ``_cdkm_iterate`` that can evaluate ``phi(i,l)``
   against

       restricted : the two clusters involved in the split (what core.py does)
       topl  L    : those two plus the L nearest centroids of the split
                    partition (the variant Section 4.2 of the paper proposes)
       fullk      : every current cluster (what Algorithm 2 in the paper
                    describes, and what the reviewer's O(T|E_j|kd) bound
                    assumes)

   The three share one implementation so their timings are directly comparable.
"""
from __future__ import annotations

import numpy as np

import noraivf.core as core
from noraivf.core import NovaIVF
from noraivf.lean import IntMap, lean_index_init


class LeanNovaIVF(NovaIVF):
    """NovaIVF whose PPM is two int32 arrays instead of two dicts."""

    def __init__(self, data, nlist, split_factor=2.0, merge_factor=0.25,
                 seed=42, use_scdkm=True):
        boot = data
        lean_index_init(self, boot, nlist, seed)

        self.split_factor = split_factor
        self.merge_factor = merge_factor
        self.use_scdkm = use_scdkm

        vecs = boot.vecs
        self.n_j = np.array([len(pl) for pl in self.posting_lists], dtype=np.int64)
        self.a_j = np.zeros((nlist, self.d), dtype=np.float32)
        self.s_j = np.zeros(nlist, dtype=np.float64)
        for c in range(nlist):
            pl = self.posting_lists[c]
            if pl:
                vc = vecs.gather(pl)
                self.a_j[c] = vc.sum(axis=0)
                self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))

        self.ppm = IntMap(vecs._cap)
        self.positions = IntMap(vecs._cap)
        for c in range(nlist):
            for off, idx in enumerate(self.posting_lists[c]):
                self.ppm[idx] = c
                self.positions[idx] = off

        self.avg_size = self.n / max(self.k, 1)
        self.T_split = self.avg_size * split_factor
        self.T_merge = self.avg_size * merge_factor
        self.L = 5


class CDKMVariant:
    """Mixin supplying a parameterised ``_cdkm_iterate``.

    ``cdkm_mode`` / ``cdkm_L`` are read off the instance so a single class can
    be swept over the three variants without rebuilding the index.
    """

    cdkm_mode = "restricted"
    cdkm_L = 5
    #: set by the harness: list of (n_candidates, n_targets, seconds)
    cdkm_log = None

    def _cdkm_targets(self, c1, c2):
        mode = self.cdkm_mode
        if mode == "restricted":
            return (c1, c2)
        if mode == "fullk":
            return range(self.k)
        if mode == "topl":
            k = self.k
            L = min(self.cdkm_L, max(k - 2, 0))
            # Nearest centroids of the split partition, by squared L2.  One
            # O(kd) pass per split, not per point.
            diff = self.centroids[:k] - self.centroids[c1].reshape(1, -1)
            d2 = np.einsum("ij,ij->i", diff, diff)
            d2[c1] = np.inf
            d2[c2] = np.inf
            near = np.argpartition(d2, L)[:L] if L < k else np.arange(k)
            out = [c1, c2]
            for c in near:
                c = int(c)
                if c != c1 and c != c2:
                    out.append(c)
            return out
        raise ValueError(f"unknown cdkm mode {mode!r}")

    def _cdkm_iterate(self, c1, c2, candidate_indices, n_iter=5):
        """Same body as core.NovaIVF._cdkm_iterate, generalised target set."""
        import time as _t

        t0 = _t.perf_counter()
        targets = self._cdkm_targets(c1, c2)

        for _ in range(n_iter):
            changed = False
            for idx in list(candidate_indices):
                if idx not in self.id_to_vec or idx not in self.ppm:
                    continue
                p = self.ppm[idx]
                vec = self.id_to_vec[idx]
                rho_i = float(np.dot(vec, vec))

                best_c = p
                best_phi = -np.inf
                for c in targets:
                    if c == p:
                        if self.n_j[c] <= 1:
                            phi = 0
                        else:
                            s_reduced = self.s_j[c] - 2 * float(np.dot(vec, self.a_j[c])) + rho_i
                            phi = self.s_j[c] / self.n_j[c] - s_reduced / (self.n_j[c] - 1)
                    else:
                        s_augmented = self.s_j[c] + 2 * float(np.dot(vec, self.a_j[c])) + rho_i
                        phi = s_augmented / (self.n_j[c] + 1) - self.s_j[c] / self.n_j[c]
                    if phi > best_phi:
                        best_phi = phi
                        best_c = c

                if best_c != p:
                    self._move_vector(idx, p, best_c, vec)
                    changed = True
            if not changed:
                break

        if self.cdkm_log is not None:
            self.cdkm_log.append(dict(
                mode=self.cdkm_mode,
                n_candidates=len(candidate_indices),
                n_targets=len(targets) if not isinstance(targets, range) else self.k,
                seconds=_t.perf_counter() - t0,
            ))


class LeanNovaIVFVariants(CDKMVariant, LeanNovaIVF):
    """LeanNovaIVF with a selectable CDKM scan width."""
