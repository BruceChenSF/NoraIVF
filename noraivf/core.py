"""
NovaIVF Experiment Suite — Core IVF Index Implementations

Implements all baseline methods and NovaIVF as described in the paper:
- Frozen, Rebuild, UpdateCentroids, DeDrift, SPFresh, NovaIVF
- Ablation variants: SCDKM-only, Mem-only
"""
import numpy as np
import faiss
import time
from abc import ABC, abstractmethod
from collections import defaultdict


def kmeans_faiss(data, k, niter=25, seed=42):
    """Run k-means using faiss, return (centroids, assignments)."""
    d = data.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed, gpu=False)
    kmeans.train(data)
    centroids = kmeans.centroids.copy()
    _, assignments = kmeans.index.search(data, 1)
    return centroids, assignments.ravel()


def brute_force_knn(queries, database, k=100):
    """Brute-force KNN using faiss."""
    d = database.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product for normalized vectors
    index.add(database)
    D, I = index.search(queries, k)
    return I


class IVFIndex(ABC):
    """Abstract base for all IVF index implementations."""

    def __init__(self, data, nlist, seed=42):
        self.d = data.shape[1]
        self.nlist = nlist
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Initial k-means
        self.centroids, self.assignments = kmeans_faiss(data, nlist, seed=seed)
        self.n = len(data)
        self.k = nlist

        # Build posting lists: list of lists of indices
        self.posting_lists = [[] for _ in range(nlist)]
        for i, c in enumerate(self.assignments):
            self.posting_lists[c].append(i)

        self.data = data.copy()
        self.id_to_vec = {i: data[i].copy() for i in range(len(data))}
        self.next_id = len(data)

        # Track metrics
        self.total_insert_time = 0.0
        self.total_delete_time = 0.0
        self.total_split_time = 0.0
        self.total_merge_time = 0.0
        self.n_inserts = 0
        self.n_deletes = 0
        self.n_splits = 0
        self.n_merges = 0
        self.n_reassigns = 0  # NPA reassignments during LIRE
        self.split_history = []  # list of (old_c, new_c, centroid_old, centroid_new) from splits

    def _centroid(self, c):
        """Get centroid of cluster c."""
        return self.centroids[c]

    def _cluster_size(self, c):
        return len(self.posting_lists[c])

    def _compute_centroid(self, c):
        """Recompute centroid from posting list data."""
        if len(self.posting_lists[c]) == 0:
            return self.centroids[c].copy()
        indices = self.posting_lists[c]
        vecs = np.array([self.id_to_vec[i] for i in indices])
        return vecs.mean(axis=0)

    def insert(self, vec):
        """Insert a vector, return its id."""
        vec = vec.astype(np.float32)
        i = self.next_id
        self.next_id += 1
        self.id_to_vec[i] = vec.copy()
        self.n += 1
        t0 = time.perf_counter()
        c = self._find_nearest_centroid(vec)
        self.posting_lists[c].append(i)
        self._post_insert(c, i, vec)
        dt = time.perf_counter() - t0
        self.total_insert_time += dt
        self.n_inserts += 1
        return i

    def delete(self, i):
        """Delete a vector by id."""
        t0 = time.perf_counter()
        c = self._find_cluster(i)
        if c is not None:
            self.posting_lists[c].remove(i)
            self._post_delete(c, i, self.id_to_vec[i])
        del self.id_to_vec[i]
        self.n -= 1
        dt = time.perf_counter() - t0
        self.total_delete_time += dt
        self.n_deletes += 1

    def search(self, query, k=100, nprobe=8):
        """Search k nearest neighbors."""
        query = query.astype(np.float32)
        # Find nearest centroids
        centroid_dists = np.dot(query, self.centroids.T)
        candidate_clusters = np.argsort(-centroid_dists)[:nprobe]

        # Scan posting lists
        heap = []  # (distance, id) — max heap of size k
        for c in candidate_clusters:
            for idx in self.posting_lists[c]:
                if idx in self.id_to_vec:
                    d = float(np.dot(query, self.id_to_vec[idx]))
                    if len(heap) < k:
                        heap.append((d, idx))
                        heap.sort(key=lambda x: x[0])  # ascending
                    elif d > heap[0][0]:
                        heap[0] = (d, idx)
                        heap.sort(key=lambda x: x[0])

        return [idx for _, idx in heap]

    def _find_nearest_centroid(self, vec):
        dists = np.dot(vec, self.centroids.T)
        return int(np.argmax(dists))

    def _find_cluster(self, vid):
        """Find which cluster a vector ID belongs to."""
        for c, pl in enumerate(self.posting_lists):
            if vid in pl:
                return c
        return None

    @abstractmethod
    def _post_insert(self, c, i, vec):
        pass

    @abstractmethod
    def _post_delete(self, c, i, vec):
        pass

    @property
    def cumul_maintenance_time(self):
        return self.total_split_time + self.total_merge_time


# ────────────────────────────────────────────────────────────────
# Frozen: completely static, no updates to centroids
# ────────────────────────────────────────────────────────────────
class FrozenIVF(IVFIndex):
    def _post_insert(self, c, i, vec):
        pass

    def _post_delete(self, c, i, vec):
        pass


# ────────────────────────────────────────────────────────────────
# Rebuild: periodic full k-means rebuild
# ────────────────────────────────────────────────────────────────
class RebuildIVF(IVFIndex):
    def __init__(self, data, nlist, rebuild_every=1000, seed=42):
        super().__init__(data, nlist, seed)
        self.rebuild_every = rebuild_every
        self.ops_since_rebuild = 0

    def _post_insert(self, c, i, vec):
        self.ops_since_rebuild += 1
        if self.ops_since_rebuild >= self.rebuild_every:
            self._rebuild()

    def _post_delete(self, c, i, vec):
        self.ops_since_rebuild += 1
        if self.ops_since_rebuild >= self.rebuild_every:
            self._rebuild()

    def _rebuild(self):
        t0 = time.perf_counter()
        if self.n < self.k * 2:
            self.ops_since_rebuild = 0
            return
        vecs = np.array([self.id_to_vec[i] for i in self.id_to_vec])
        self.centroids, self.assignments = kmeans_faiss(vecs, self.nlist, seed=self.seed)
        self.posting_lists = [[] for _ in range(self.nlist)]
        for i, c in zip(self.id_to_vec.keys(), self.assignments):
            self.posting_lists[c].append(i)
        self.ops_since_rebuild = 0
        dt = time.perf_counter() - t0
        self.total_split_time += dt
        self.n_splits += 1


# ────────────────────────────────────────────────────────────────
# UpdateCentroids: update centroids on insert, no split/merge
# ────────────────────────────────────────────────────────────────
class UpdateCentroidsIVF(IVFIndex):
    def __init__(self, data, nlist, seed=42):
        super().__init__(data, nlist, seed)
        self.cluster_sums = [None] * nlist
        for c in range(nlist):
            if len(self.posting_lists[c]) > 0:
                indices = self.posting_lists[c]
                self.cluster_sums[c] = np.array(
                    [self.id_to_vec[i] for i in indices]
                ).sum(axis=0)
            else:
                self.cluster_sums[c] = np.zeros(self.d, dtype=np.float32)

    def _post_insert(self, c, i, vec):
        if self.cluster_sums[c] is None:
            self.cluster_sums[c] = vec.copy()
        else:
            self.cluster_sums[c] += vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz

    def _post_delete(self, c, i, vec):
        if self.cluster_sums[c] is not None:
            self.cluster_sums[c] -= vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz


# ────────────────────────────────────────────────────────────────
# SPFresh (memory-ported): 2-means split + LIRE reassignment
# ────────────────────────────────────────────────────────────────
class SPFreshIVF(IVFIndex):
    def __init__(self, data, nlist, split_factor=2.0, merge_factor=0.25, 
                 seed=42, do_reassign=True):
        super().__init__(data, nlist, seed)
        self.split_factor = split_factor
        self.merge_factor = merge_factor
        self.do_reassign = do_reassign  # False = skip LIRE NPA reassignment
        self.cluster_sums = [None] * nlist
        for c in range(nlist):
            indices = self.posting_lists[c]
            if len(indices) > 0:
                self.cluster_sums[c] = np.array(
                    [self.id_to_vec[i] for i in indices]
                ).sum(axis=0).astype(np.float32)
            else:
                self.cluster_sums[c] = np.zeros(self.d, dtype=np.float32)
        self.avg_size = self.n / max(self.k, 1)
        self.T_split = self.avg_size * self.split_factor
        self.T_merge = self.avg_size * self.merge_factor
        self.L = 5  # LIRE neighborhood size

    def _post_insert(self, c, i, vec):
        self.cluster_sums[c] += vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz
        if sz > self.T_split:
            self._lire_split(c)

    def _post_delete(self, c, i, vec):
        self.cluster_sums[c] -= vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz
        if sz < self.T_merge and sz > 0:
            self._merge(c)

    def _lire_split(self, j):
        """SPFresh LIRE Split: 2-means + LIRE reassignment."""
        t0 = time.perf_counter()
        if self._cluster_size(j) < 4:
            return

        # Get vectors in cluster j
        indices_j = self.posting_lists[j]
        vecs_j = np.array([self.id_to_vec[i] for i in indices_j], dtype=np.float32)

        # 2-means clustering
        centroids_2, assigns_2 = kmeans_faiss(vecs_j, 2, niter=10, seed=self.rng.randint(0, 2**31))

        # Create new cluster
        new_c = self.k
        self.k += 1
        self.posting_lists.append([])
        self.cluster_sums.append(np.zeros(self.d, dtype=np.float32))
        cen0 = centroids_2[0].copy()
        cen1 = centroids_2[1].copy()

        # Assign vectors
        new_assigns_j = []
        new_assigns_new = []
        for idx, a in zip(indices_j, assigns_2):
            if a == 0:
                new_assigns_j.append(idx)
            else:
                new_assigns_new.append(idx)
        self.posting_lists[j] = new_assigns_j
        self.posting_lists[new_c] = new_assigns_new

        # Update centroids
        if len(new_assigns_j) > 0:
            self.centroids[j] = np.array([self.id_to_vec[i] for i in new_assigns_j]).mean(axis=0)
            self.cluster_sums[j] = np.array([self.id_to_vec[i] for i in new_assigns_j]).sum(axis=0)
        if len(new_assigns_new) > 0:
            self.centroids = np.vstack([self.centroids, cen1.reshape(1, -1)])
            self.cluster_sums[new_c] = np.array([self.id_to_vec[i] for i in new_assigns_new]).sum(axis=0)
        else:
            self.centroids = np.vstack([self.centroids, np.zeros((1, self.d), dtype=np.float32)])

        # LIRE protocol: scan j and new_c + top-L neighbors, fix NPA violations
        if self.do_reassign:
            centroid_dists = np.dot(self.centroids[j], self.centroids.T)
            centroid_dists[j] = -np.inf
            centroid_dists[new_c] = -np.inf
            L = min(self.L, self.k - 2)
            top_L = np.argsort(-centroid_dists)[:L]

            affected = set([j, new_c] + list(top_L))
            n_reassigned = 0
            for c in affected:
                for idx in list(self.posting_lists[c]):
                    if idx not in self.id_to_vec:
                        continue
                    vec = self.id_to_vec[idx]
                    dists = np.dot(vec, self.centroids[:self.k].T)
                    best_c = int(np.argmax(dists))
                    if best_c != c:
                        self.posting_lists[c].remove(idx)
                        self.posting_lists[best_c].append(idx)
                        self.cluster_sums[c] -= vec
                        self.cluster_sums[best_c] += vec
                        n_reassigned += 1

            self.n_reassigns += n_reassigned

            # Update affected centroids
            for c in affected:
                sz = self._cluster_size(c)
                if sz > 0:
                    self.centroids[c] = self.cluster_sums[c] / sz
        else:
            # No LIRE reassignment — just 2-means split with no boundary fix
            pass

        self.n_splits += 1
        self.split_history.append((j, new_c, self.centroids[j].copy(), self.centroids[new_c].copy()))

        dt = time.perf_counter() - t0
        self.total_split_time += dt

    def _merge(self, j):
        """Merge small cluster j into nearest neighbor."""
        t0 = time.perf_counter()
        if self._cluster_size(j) == 0:
            return
        # Find nearest cluster
        centroid_dists = np.dot(self.centroids[j], self.centroids.T)
        centroid_dists[j] = -np.inf
        m = int(np.argmax(centroid_dists))

        # Move vectors
        for idx in self.posting_lists[j]:
            self.posting_lists[m].append(idx)
        self.cluster_sums[m] += self.cluster_sums[j]
        self.posting_lists[j] = []
        self.cluster_sums[j] = np.zeros(self.d, dtype=np.float32)

        if self._cluster_size(m) > 0:
            self.centroids[m] = self.cluster_sums[m] / self._cluster_size(m)

        self.n_merges += 1
        dt = time.perf_counter() - t0
        self.total_merge_time += dt


# ────────────────────────────────────────────────────────────────
# NovaIVF: SCDKM protocol + ClusterEntry + PostingPointerMap
# ────────────────────────────────────────────────────────────────
class NovaIVF(IVFIndex):
    def __init__(self, data, nlist, split_factor=2.0, merge_factor=0.25, seed=42, use_scdkm=True):
        super().__init__(data, nlist, seed)
        self.split_factor = split_factor
        self.merge_factor = merge_factor
        self.use_scdkm = use_scdkm

        # ClusterEntry variables: n_j, s_j, a_j (vector sum)
        self.n_j = np.array([len(pl) for pl in self.posting_lists], dtype=np.int64)
        self.a_j = np.zeros((nlist, self.d), dtype=np.float32)  # sum of vectors
        self.s_j = np.zeros(nlist, dtype=np.float64)  # squared norm of a_j

        for c in range(nlist):
            if len(self.posting_lists[c]) > 0:
                vecs_c = np.array([self.id_to_vec[i] for i in self.posting_lists[c]])
                self.a_j[c] = vecs_c.sum(axis=0)
                self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))

        # PostingPointerMap: id -> (cluster, offset)
        self.ppm = {}  # id -> cluster_id
        self.positions = {}  # id -> offset in posting list (for swap-and-pop)
        for c in range(nlist):
            for off, idx in enumerate(self.posting_lists[c]):
                self.ppm[idx] = c
                self.positions[idx] = off

        # Compact posting lists as numpy arrays where possible
        # Keep as Python lists for simplicity but track positions

        self.avg_size = self.n / max(self.k, 1)
        self.T_split = self.avg_size * self.split_factor
        self.T_merge = self.avg_size * self.merge_factor
        self.L = 5

    def insert(self, vec):
        vec = vec.astype(np.float32)
        i = self.next_id
        self.next_id += 1
        self.id_to_vec[i] = vec.copy()
        self.n += 1
        self.n_inserts += 1

        t0 = time.perf_counter()
        # Find nearest centroid
        dists = np.dot(vec, self.centroids.T)
        c = int(np.argmax(dists))

        # Append to posting list
        off = len(self.posting_lists[c])
        self.posting_lists[c].append(i)
        self.ppm[i] = c
        self.positions[i] = off

        # Update CDKM variables
        self.n_j[c] += 1
        self.a_j[c] += vec
        self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))
        if self.n_j[c] > 0:
            self.centroids[c] = self.a_j[c] / self.n_j[c]

        dt = time.perf_counter() - t0
        self.total_insert_time += dt

        # Check split
        if self.n_j[c] > self.T_split:
            self._split(c)

        return i

    def delete(self, i):
        t0 = time.perf_counter()
        c = self.ppm.get(i)
        if c is None:
            return
        vec = self.id_to_vec[i]
        pl = self.posting_lists[c]

        # Find position — may be stale after splits, fix by scanning
        off = self.positions.get(i, -1)
        if off < 0 or off >= len(pl) or pl[off] != i:
            # Position is stale, find by linear scan
            try:
                off = pl.index(i)
            except ValueError:
                del self.id_to_vec[i]
                return

        # Swap-and-pop
        last_idx = pl[-1]
        pl[off] = last_idx
        self.positions[last_idx] = off
        pl.pop()

        # Update CDKM variables
        self.n_j[c] -= 1
        self.a_j[c] -= vec
        self.s_j[c] = float(np.dot(self.a_j[c], self.a_j[c]))
        if self.n_j[c] > 0:
            self.centroids[c] = self.a_j[c] / self.n_j[c]

        del self.ppm[i]
        del self.positions[i]
        del self.id_to_vec[i]
        self.n -= 1
        self.n_deletes += 1

        dt = time.perf_counter() - t0
        self.total_delete_time += dt

        # Check merge
        if self.n_j[c] < self.T_merge and self.n_j[c] > 0:
            self._merge(c)

    def _split(self, j):
        """SCDKM Split using CDKM-based local optimization."""
        t0 = time.perf_counter()
        if self.n_j[j] < 4:
            return

        self.n_splits += 1

        # Get vectors in cluster j
        indices_j = list(self.posting_lists[j])
        vecs_j = np.array([self.id_to_vec[i] for i in indices_j], dtype=np.float32)

        # Random initial split
        new_c = self.k
        self.k += 1
        perm = self.rng.permutation(len(indices_j))
        half = len(indices_j) // 2
        assigns_j = indices_j[:half] if half > 0 else []
        assigns_new = indices_j[half:] if half < len(indices_j) else []

        self.posting_lists[j] = list(assigns_j)
        self.posting_lists.append(list(assigns_new))

        # Compute initial CDKM variables for new cluster
        if len(assigns_new) > 0:
            vecs_new = np.array([self.id_to_vec[i] for i in assigns_new])
            a_new = vecs_new.sum(axis=0)
            n_new = len(assigns_new)
            s_new = float(np.dot(a_new, a_new))
        else:
            a_new = np.zeros(self.d, dtype=np.float32)
            n_new = 0
            s_new = 0.0

        # Update old cluster variables
        if len(assigns_j) > 0:
            vecs_j_rem = np.array([self.id_to_vec[i] for i in assigns_j])
            self.a_j[j] = vecs_j_rem.sum(axis=0)
            self.n_j[j] = len(assigns_j)
            self.s_j[j] = float(np.dot(self.a_j[j], self.a_j[j]))
            self.centroids[j] = self.a_j[j] / self.n_j[j]
        else:
            self.a_j[j] = np.zeros(self.d, dtype=np.float32)
            self.n_j[j] = 0
            self.s_j[j] = 0.0

        # Extend arrays
        self.n_j = np.append(self.n_j, n_new)
        self.s_j = np.append(self.s_j, s_new)
        self.a_j = np.vstack([self.a_j, a_new.reshape(1, -1)])
        self.centroids = np.vstack([
            self.centroids,
            (a_new / n_new).reshape(1, -1) if n_new > 0 else np.zeros((1, self.d), dtype=np.float32)
        ])

        # Update PPM for moved vectors and fix positions for all
        for new_off, idx in enumerate(self.posting_lists[j]):
            self.ppm[idx] = j
            self.positions[idx] = new_off
        for new_off, idx in enumerate(self.posting_lists[new_c]):
            self.ppm[idx] = new_c
            self.positions[idx] = new_off

        if self.use_scdkm:
            # SCDKM: CDKM iterations on local vectors only
            self._cdkm_iterate(j, new_c, indices_j)
        else:
            # Mem-only mode: use LIRE reassignment (2-means already done via random split)
            # For proper LIRE, we'd do 2-means first, then LIRE
            # Here we scan affected clusters for NPA violations
            centroid_dists = np.dot(self.centroids[j], self.centroids.T)
            centroid_dists[j] = -np.inf
            centroid_dists[new_c] = -np.inf
            L = min(self.L, self.k - 2)
            top_L = np.argsort(-centroid_dists)[:L]
            affected = set([j, new_c] + list(top_L))
            n_reassigned = 0
            for c in affected:
                for idx in list(self.posting_lists[c]):
                    if idx not in self.id_to_vec:
                        continue
                    vec = self.id_to_vec[idx]
                    dists = np.dot(vec, self.centroids.T)
                    best_c = int(np.argmax(dists))
                    if best_c != c:
                        self._move_vector(idx, c, best_c, vec)
                        n_reassigned += 1
            self.n_reassigns += n_reassigned

        dt = time.perf_counter() - t0
        self.total_split_time += dt
        self.split_history.append((j, new_c, self.centroids[j].copy(), self.centroids[new_c].copy()))

    def _cdkm_iterate(self, c1, c2, candidate_indices, n_iter=5):
        """Run CDKM iterations on local vectors (SCDKM protocol)."""
        for _ in range(n_iter):
            changed = False
            for idx in list(candidate_indices):
                if idx not in self.id_to_vec or idx not in self.ppm:
                    continue
                p = self.ppm[idx]
                vec = self.id_to_vec[idx]
                rho_i = float(np.dot(vec, vec))

                # Compute phi for each cluster
                best_c = p
                best_phi = -np.inf
                for c in [c1, c2]:
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

    def _move_vector(self, idx, from_c, to_c, vec):
        """Move a vector between clusters with full bookkeeping."""
        # Remove from source
        pl_from = self.posting_lists[from_c]
        off = self.positions[idx]
        if off < len(pl_from) - 1:
            last_idx = pl_from[-1]
            pl_from[off] = last_idx
            self.positions[last_idx] = off
        pl_from.pop()

        self.n_j[from_c] -= 1
        self.a_j[from_c] -= vec
        self.s_j[from_c] = float(np.dot(self.a_j[from_c], self.a_j[from_c]))
        if self.n_j[from_c] > 0:
            self.centroids[from_c] = self.a_j[from_c] / self.n_j[from_c]

        # Add to destination
        pl_to = self.posting_lists[to_c]
        new_off = len(pl_to)
        pl_to.append(idx)
        self.ppm[idx] = to_c
        self.positions[idx] = new_off

        self.n_j[to_c] += 1
        self.a_j[to_c] += vec
        self.s_j[to_c] = float(np.dot(self.a_j[to_c], self.a_j[to_c]))
        self.centroids[to_c] = self.a_j[to_c] / self.n_j[to_c]

    def _merge(self, j):
        """Merge cluster j into nearest neighbor with swap-with-last for ClusterEntry list."""
        t0 = time.perf_counter()
        if self.n_j[j] == 0:
            return
        self.n_merges += 1

        # Find nearest cluster
        centroid_dists = np.dot(self.centroids[j], self.centroids.T)
        centroid_dists[j] = -np.inf
        m = int(np.argmax(centroid_dists))

        # Move vectors from j to m
        old_len_m = len(self.posting_lists[m])
        for off, idx in enumerate(list(self.posting_lists[j])):
            self.posting_lists[m].append(idx)
            self.ppm[idx] = m
            self.positions[idx] = old_len_m + off

        # Update m's CDKM variables
        vec = self.id_to_vec
        self.n_j[m] += self.n_j[j]
        self.a_j[m] += self.a_j[j]
        self.s_j[m] = float(np.dot(self.a_j[m], self.a_j[m]))
        self.centroids[m] = self.a_j[m] / self.n_j[m]

        # Clear j
        self.posting_lists[j] = []
        self.n_j[j] = 0
        self.a_j[j] = np.zeros(self.d, dtype=np.float32)
        self.s_j[j] = 0.0

        dt = time.perf_counter() - t0
        self.total_merge_time += dt

    def _post_insert(self, c, i, vec):
        pass  # Handled in overridden insert()

    def _post_delete(self, c, i, vec):
        pass  # Handled in overridden delete()


# ────────────────────────────────────────────────────────────────
# DeDrift: k2-means rebalancing on every update
# ────────────────────────────────────────────────────────────────
class DeDriftIVF(IVFIndex):
    def __init__(self, data, nlist, k2=10, rebalance_every=100, seed=42):
        super().__init__(data, nlist, seed)
        self.k2 = k2
        self.rebalance_every = rebalance_every
        self.ops_since_rebalance = 0
        self.cluster_sums = [None] * nlist
        for c in range(nlist):
            if len(self.posting_lists[c]) > 0:
                indices = self.posting_lists[c]
                self.cluster_sums[c] = np.array(
                    [self.id_to_vec[i] for i in indices]).sum(axis=0).astype(np.float32)
            else:
                self.cluster_sums[c] = np.zeros(self.d, dtype=np.float32)

    def _post_insert(self, c, i, vec):
        self.cluster_sums[c] += vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz
        self.ops_since_rebalance += 1
        if self.ops_since_rebalance >= self.rebalance_every:
            self._rebalance()
            self.ops_since_rebalance = 0

    def _post_delete(self, c, i, vec):
        self.cluster_sums[c] -= vec
        sz = self._cluster_size(c)
        if sz > 0:
            self.centroids[c] = self.cluster_sums[c] / sz
        self._rebalance()

    def _rebalance(self):
        """DeDrift rebalancing: split k largest, merge with k2-k smallest, k2-means."""
        t0 = time.perf_counter()
        sizes = [(c, len(pl)) for c, pl in enumerate(self.posting_lists) if len(pl) > 0]
        if len(sizes) < 5:
            return
        sizes.sort(key=lambda x: -x[1])
        k = min(self.k2 // 2, len(sizes) // 2)
        largest = sizes[:k]
        smallest = sizes[-k:]

        # Collect vectors from largest + smallest clusters
        all_indices = []
        for c, _ in largest + smallest:
            all_indices.extend(self.posting_lists[c])
        if len(all_indices) < self.k2 * 2:
            return
        vecs = np.array([self.id_to_vec[i] for i in all_indices], dtype=np.float32)
        centroids_new, assigns = kmeans_faiss(vecs, self.k2, niter=10, seed=self.rng.randint(0, 2**31))

        # Reassign to these k2 clusters
        for c_orig, _ in largest + smallest:
            self.posting_lists[c_orig] = []
            self.cluster_sums[c_orig] = np.zeros(self.d, dtype=np.float32)
            self.centroids[c_orig] = np.zeros(self.d, dtype=np.float32)

        for c_target in range(self.k2):
            c_actual = largest[c_target % len(largest)][0] if c_target < len(largest) else smallest[c_target - len(largest)][0]
            idxs = [all_indices[i] for i, a in enumerate(assigns) if a == c_target]
            self.posting_lists[c_actual] = idxs
            if idxs:
                self.cluster_sums[c_actual] = np.array([self.id_to_vec[i] for i in idxs]).sum(axis=0)
                self.centroids[c_actual] = self.cluster_sums[c_actual] / len(idxs)

        self.n_splits += 1
        dt = time.perf_counter() - t0
        self.total_split_time += dt


# ────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────
def create_index(name, data, nlist, **kwargs):
    """Create an IVF index by name."""
    name = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if name == "frozen":
        return FrozenIVF(data, nlist, **kwargs)
    elif name == "rebuild":
        return RebuildIVF(data, nlist, **kwargs)
    elif name in ("updatecentroids", "update_centroids"):
        return UpdateCentroidsIVF(data, nlist, **kwargs)
    elif name == "dedrift":
        return DeDriftIVF(data, nlist, **kwargs)
    elif name in ("spfresh",):
        return SPFreshIVF(data, nlist, **kwargs)
    elif name in ("novaivf",):
        seed = kwargs.pop("seed", 42)
        return NovaIVF(data, nlist, seed=seed, use_scdkm=True, **kwargs)
    elif name in ("scdkm", "scdkmonly", "scdkm_only"):
        seed = kwargs.pop("seed", 42)
        return NovaIVF(data, nlist, seed=seed, use_scdkm=True, **kwargs)
    elif name in ("memonly", "mem_only"):
        seed = kwargs.pop("seed", 42)
        return NovaIVF(data, nlist, seed=seed, use_scdkm=False, **kwargs)
    else:
        raise ValueError(f"Unknown index: {name}")
