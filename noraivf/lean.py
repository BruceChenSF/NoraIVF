"""Memory-lean substrate for running the reference NoraIVF protocols at 1e9 scale.

The reference implementation in ``noraivf/core.py`` is a faithful but
memory-hungry simulator: ``IVFIndex.__init__`` materialises

    self.data      = data.copy()                       # 512 B/vector at d=128 f32
    self.id_to_vec = {i: data[i].copy() for i in ...}   # + a numpy object per vector

which at 1e9 vectors costs well over 700 GB before any index structure exists.
The algorithms themselves, however, only ever touch ``id_to_vec`` through the
plain mapping protocol (``x[i]``, ``x[i] = v``, ``del x[i]``, ``i in x``,
``for i in x``, ``.keys()``, ``.items()``) and ``ppm``/``positions`` through the
same protocol plus ``.get``.

So instead of re-implementing six maintenance protocols -- which would make the
billion-scale numbers incomparable with the published 1M/10M numbers -- this
module keeps the protocols byte-identical and swaps only the *containers*:

    VecStore  : dict-like id -> float32 vector, backed by one contiguous array
    IntMap    : dict-like id -> int, backed by one int32 array

Everything else (``posting_lists`` as Python lists, the split/merge routines,
the CDKM iterations, the NERROR bookkeeping) is the reference code, executed
unchanged.  Per-vector state drops from ~1.28 kB to ~0.31 kB, which is what
makes 1e9 possible on a 503 GB machine.

Storage dtype
-------------
Vectors are stored in ``float16`` by default.  SIFT components are integers in
0..255 and float16 represents every integer up to 2048 exactly, so the float32
values handed back by ``VecStore.__getitem__`` are *bit-identical* to the ones
the reference would have stored; the only cost is the widening conversion on
read.  ``--storage float32`` is available for data that needs it.
"""
from __future__ import annotations

import time

import numpy as np

import noraivf.core as core


# ────────────────────────────────────────────────────────────────
# dict-like containers
# ────────────────────────────────────────────────────────────────
class VecStore:
    """id -> float32 vector, backed by one contiguous ``(cap, d)`` array."""

    __slots__ = ("_buf", "_alive", "_cap", "_d", "_n", "_widen", "_dtype")

    def __init__(self, capacity, dim, storage_dtype="float16"):
        self._cap = int(capacity)
        self._d = int(dim)
        self._dtype = np.dtype(storage_dtype)
        self._widen = self._dtype != np.dtype(np.float32)
        self._buf = np.empty((self._cap, self._d), dtype=self._dtype)
        self._alive = np.zeros(self._cap, dtype=bool)
        self._n = 0

    # --- plain mapping protocol -----------------------------------
    def __getitem__(self, i):
        row = self._buf[i]
        # A float32 row is returned as a view (the reference returned a copy;
        # no caller mutates the value it reads); narrower storage is widened,
        # which is exact for the datasets this harness is used on.
        return row.astype(np.float32) if self._widen else row

    def __setitem__(self, i, v):
        self._buf[i] = v
        if not self._alive[i]:
            self._alive[i] = True
            self._n += 1

    def __delitem__(self, i):
        if self._alive[i]:
            self._alive[i] = False
            self._n -= 1

    def __contains__(self, i):
        return 0 <= i < self._cap and bool(self._alive[i])

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(np.nonzero(self._alive)[0])

    def keys(self):
        return np.nonzero(self._alive)[0]

    def items(self):
        for i in np.nonzero(self._alive)[0]:
            yield int(i), self[i]

    def get(self, i, default=None):
        return self[i] if i in self else default

    # --- bulk helpers used by the harness -------------------------
    def load_from(self, src, n, chunk=4_000_000):
        """Copy the first ``n`` rows of a (possibly memmapped) array in."""
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            self._buf[s:e] = src[s:e]
        self._alive[:n] = True
        self._n = n

    def gather(self, ids):
        """Vectorised gather of a sequence of ids -> (m, d) float32."""
        idx = np.asarray(ids, dtype=np.int64)
        out = self._buf[idx]
        return out.astype(np.float32) if self._widen else out

    def resident_bytes(self):
        return self._buf.nbytes + self._alive.nbytes


class IntMap:
    """id -> non-negative int, backed by an int32 array (``-1`` = absent)."""

    __slots__ = ("_a", "_fill")

    def __init__(self, capacity, fill=-1, dtype=np.int32):
        self._fill = fill
        self._a = np.full(int(capacity), fill, dtype=dtype)

    def __getitem__(self, k):
        return int(self._a[k])

    def __setitem__(self, k, v):
        self._a[k] = v

    def __delitem__(self, k):
        self._a[k] = self._fill

    def __contains__(self, k):
        return 0 <= k < self._a.shape[0] and int(self._a[k]) != self._fill

    def get(self, k, default=None):
        if 0 <= k < self._a.shape[0]:
            v = int(self._a[k])
            if v != self._fill:
                return v
        return default

    def __len__(self):
        return int(np.count_nonzero(self._a != self._fill))

    def resident_bytes(self):
        return self._a.nbytes


# ────────────────────────────────────────────────────────────────
# lean IVFIndex.__init__
# ────────────────────────────────────────────────────────────────
class Bootstrap:
    """Everything ``IVFIndex.__init__`` needs, without touching the raw data.

    Passed to the index constructors *in place of* the ``data`` argument, so the
    reference ``__init__`` signatures and all six subclass ``__init__``s run
    unchanged.
    """

    __slots__ = ("vecs", "centroids", "assignments", "n", "n_init", "timing")

    def __init__(self, vecs, centroids, assignments, n_init, timing=None):
        self.vecs = vecs
        self.centroids = centroids
        self.assignments = assignments
        self.n_init = int(n_init)
        self.n = int(n_init)
        self.timing = timing if timing is not None else {}

    def __len__(self):
        return self.n_init

    @property
    def shape(self):
        return (self.n_init, self.vecs._d)


def lean_index_init(self, data, nlist, seed=42):
    """Drop-in replacement for ``IVFIndex.__init__`` (monkey-patched in)."""
    boot = data
    t0 = time.perf_counter()

    self.d = boot.vecs._d
    self.nlist = nlist
    self.seed = seed
    self.rng = np.random.RandomState(seed)

    self.vecs = boot.vecs
    self.centroids = boot.centroids
    self.assignments = boot.assignments
    self.n = boot.n_init
    self.k = nlist

    self.posting_lists = _build_postings(boot.assignments, nlist)
    boot.timing["postings_s"] = time.perf_counter() - t0

    # The reference sets `self.data = data.copy()` (never read) and
    # `self.id_to_vec = {i: ...}`; here both are views onto the shared store.
    self.data = None
    self.id_to_vec = boot.vecs
    self.next_id = boot.n_init

    self.total_insert_time = 0.0
    self.total_delete_time = 0.0
    self.total_split_time = 0.0
    self.total_merge_time = 0.0
    self.n_inserts = 0
    self.n_deletes = 0
    self.n_splits = 0
    self.n_merges = 0
    self.n_reassigns = 0
    self.split_history = []


def _build_postings(assignments, nlist):
    """Group ids by cluster, preserving ascending id order (as the reference
    append loop does)."""
    order = np.argsort(assignments, kind="stable")
    counts = np.bincount(assignments, minlength=nlist)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [order[bounds[c]:bounds[c + 1]].tolist() for c in range(nlist)]


def install():
    """Monkey-patch ``IVFIndex.__init__``.  Idempotent."""
    if getattr(core.IVFIndex.__init__, "_lean", False):
        return
    lean_index_init._lean = True
    core.IVFIndex.__init__ = lean_index_init


# ────────────────────────────────────────────────────────────────
# build helpers
# ────────────────────────────────────────────────────────────────
def kmeans_on_sample(vecs, n_init, nlist, seed=42, niter=25, sample=None):
    """Train a coarse quantizer on a subsample.

    When the corpus fits in the sample budget every point is used **in natural
    order**, which reproduces ``kmeans_faiss(data, nlist, seed)`` exactly (faiss
    draws its initial centres by index, so the array order is part of the
    result).  Only when the corpus is larger do we draw a random subsample,
    which is what faiss itself does once ``n > nlist * max_points_per_centroid``.
    """
    import faiss

    if sample is None:
        sample = max(nlist * 256, 1_000_000)
    if n_init <= sample:
        idx = np.arange(n_init, dtype=np.int64)
        sampled = False
    else:
        rng = np.random.RandomState(seed)
        idx = np.sort(np.unique(rng.randint(0, n_init, size=int(sample * 1.05) + 16)))[:sample]
        sampled = True
    X = vecs.gather(idx)
    km = faiss.Kmeans(vecs._d, nlist, niter=niter, verbose=False, seed=seed, gpu=False)
    km.train(X)
    return km.centroids.copy(), sampled, len(idx)


def assign_all(vecs, centroids, n, chunk=2_000_000, verbose_every=50):
    """Assign ``n`` vectors to their nearest centroid, in chunks."""
    import faiss

    quant = faiss.IndexFlatL2(centroids.shape[1])
    quant.add(centroids)
    out = np.empty(n, dtype=np.int32)
    t0 = time.perf_counter()
    done = 0
    while done < n:
        e = min(done + chunk, n)
        X = vecs.gather(np.arange(done, e))
        _, I = quant.search(X, 1)
        out[done:e] = I.ravel()
        done = e
        del X, I
    return out, time.perf_counter() - t0


def load_vecstore(path, n, dim, storage_dtype="float16", capacity=None, chunk=4_000_000):
    """Build a VecStore from an npy corpus (memmapped, copied in chunks)."""
    mm = np.load(path, mmap_mode="r")
    assert mm.shape[1] == dim, (mm.shape, dim)
    vs = VecStore(capacity or n, dim, storage_dtype)
    vs.load_from(mm, n, chunk=chunk)
    return vs


def verify_lossless(src, n, storage_dtype, sample=20_000, seed=0):
    """Check that the chosen storage dtype reproduces float32 exactly."""
    mm = np.load(src, mmap_mode="r")
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.randint(0, n, size=sample))
    a = np.asarray(mm[idx], dtype=np.float32)
    b = np.asarray(mm[idx]).astype(storage_dtype).astype(np.float32)
    bad = int(np.count_nonzero(a != b))
    return bad, float(np.abs(a).max()), len(idx)
