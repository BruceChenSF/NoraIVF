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
# AdaIVF: incremental maintenance port of arXiv:2411.00970 (Ada-IVF)
# ────────────────────────────────────────────────────────────────
class AdaIVF(IVFIndex):
    """Ada-IVF incremental index maintenance (arXiv:2411.00970, Algorithm 4).

    Per-partition metadata: temperature T (init 1.0, floor 1.0), running
    centroid mu (incremental update), initial-centroid snapshot mu0 (drift
    anchor), and size s.  A PAM (id -> cluster) is kept so deletes are O(1)
    (the paper uses PAM to support deletions; the base-class linear scan is
    too slow and unfair as a baseline).

    Violation detection runs every ``check_every`` update ops, vectorized:
        f(c) = alpha * T(c) * ( beta * f_s(c) + (1-beta) * f_d(c) )
    with f_s the bidirectional relative size deviation and f_d the centroid
    drift ||mu - mu0|| / ||mu0||.  Partitions with f > tau_f are violators.
    The paper tunes tau_f by workload/load; the default here (0.75) targets
    the no-query streaming phase (T=1, f = 0.5*f_s + 0.5*f_d): naturally
    sized clusters (0.25x..2.5x tau_s, normal drift) stay below 0.75 and
    only genuinely unbalanced partitions trigger.  For query-bearing loads
    (hotspot workloads) the temperature multiplies f, so the working point
    is tau_f ~= 1.0: on SIFT-50k (smoke, seed=42) tau_f=1.0 gives
    hot-cold recall gap +0.062 vs +0.010 with temperature ablated, while
    0.75 with temperature ON over-maintains (gap -0.037 vs -0.007 OFF).
    A global indicator G (size-std + reconstruction-error deviation) guards
    against global drift and triggers a full k-means rebuild when G > tau_g
    (default tau_g=5.0).  The paper does not specify a value (it is workload
    dependent); 5.0 is deliberately loose so the safety valve fires only in
    genuinely pathological global drift, not during normal streaming.

    Local reindex (Algorithm 4): violator partitions (and, if oversized,
    their kmeans split centroids) enter candidate pool M, all their points
    enter reclustering pool X, and their slots are released; the r_c nearest
    index centroids to M also join M/X with released slots; warm-started
    Lloyd iterations (iota=5) partition X using M as init; over-allocated
    candidate pools are trimmed before Lloyd, and after Lloyd every pool
    partition is balanced up to >= max(2, int(0.75*tau_s)) points; new partitions
    are written back into the released
    slots (appending when the pool grows).

    Documented deviations from the paper:
      * the paper uses balanced k-means for the pool; we approximate it with
        warm-started Lloyd + a post-Lloyd balancing pass (_balance) that forces
        every pool partition to >= max(2, int(0.75*tau_s)) points by moving the most
        similar points from the largest surplus partitions into under-size ones,
        plus a pre-Lloyd pool trim that drops candidate centroids the data
        cannot sustain (cheaper than a true balanced k-means, same spirit),
        with inner-product (cosine) assignment on unit-normalized vectors.
      * the paper fixes tau_s=1000 for its nlist regime; here the default is
        the equilibrium average partition size, max(8, int(round(n_init/nlist))),
        so a balanced partition sits at f_s ~ 0 and only real size deviation
        (over- or under-sized clusters) flags a violator.  The old default of
        2*n_init/nlist made f_s ~ 1.0 at equilibrium (every cluster a violator,
        degenerating to repeated full rebuilds).  Experiments may override it.
      * temperature is driven by queries: probed partitions heat as
        T *= (1 + eta*sim_nearest/sim_c) (factor clamped to [1, 2] to keep
        the per-query heating bounded), unprobed ones cool as
        T = max(T*(1-nu), 1.0).  If the streaming phase issues no queries,
        T stays 1.0 and detection degrades to pure size+drift, which is a
        reasonable degradation (the paper's temperature needs a query stream).

      * ablation switch ``use_temperature``: with False, ``search()`` skips
        the query-driven temperature updates entirely, so T stays exactly
        1.0 for the whole run and violation detection degenerates to pure
        size+drift (``f = alpha*(beta*f_s + (1-beta)*f_d)``).  This isolates
        the contribution of the temperature mechanism (query-locality
        amplification) from the rest of Algorithm 4; experiments that do not
        exercise query locality keep the default True.

    Data is L2-normalized internally (``normalize=True``) so inner product
    equals cosine similarity and the per-partition reconstruction error
    reduces to eps_c = 1 - ||mu_c||^2 (exact for unit vectors).  With
    ``normalize=False`` (raw-Euclidean workloads, see L2AdaIVF in the
    ann-benchmarks runner) the aggregate formulas remain exact because we
    maintain per-partition vector sums a_c and squared-norm sums q_c, giving
    eps_c = q_c/s_c - ||mu_c||^2 for arbitrary data.

    Maintenance time is charged to ``total_split_time`` so
    ``cumul_maintenance_time`` reports it automatically.  Extra statistics
    for experiment reporting live in ``ada_stats``.
    """

    def __init__(self, data, nlist, normalize=True, tau_s=None, alpha=1.0,
                 beta=0.5, tau_f=0.75, r_c=25, iota=5, check_every=100,
                 gamma=0.5, tau_g=5.0, eta=0.1, nu=0.01, seed=42,
                 use_temperature=True):
        # use_temperature=False: ablation — search() skips temperature
        # updates, T stays 1.0, detection reduces to pure size+drift.
        self.use_temperature = bool(use_temperature)
        self.normalize = bool(normalize)
        data = np.asarray(data, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            data = data / norms
        super().__init__(data, nlist, seed=seed)

        n_init = len(data)
        self.tau_s = tau_s if tau_s is not None else max(8, int(round(n_init / nlist)))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_f = float(tau_f)
        self.r_c = int(r_c)
        self.iota = int(iota)
        self.check_every = int(check_every)
        self.gamma = float(gamma)
        self.tau_g = float(tau_g)
        self.eta = float(eta)
        self.nu = float(nu)

        # Per-partition metadata: temperature, drift anchor, sums.
        self.T = np.ones(self.k, dtype=np.float64)
        self.mu0 = self.centroids.copy()
        self.a_c = np.zeros((self.k, self.d), dtype=np.float32)
        self.q_c = np.zeros(self.k, dtype=np.float64)
        for c in range(self.k):
            pl = self.posting_lists[c]
            if pl:
                vecs = np.array([self.id_to_vec[i] for i in pl], dtype=np.float32)
                self.a_c[c] = vecs.sum(axis=0)
                self.q_c[c] = float(np.sum(vecs * vecs))

        # PAM: id -> cluster for O(1) delete lookup.
        self.pam = {}
        for c in range(self.k):
            for i in self.posting_lists[c]:
                self.pam[i] = c

        self.ops_since_check = 0
        self.n0 = n_init
        sizes = np.array([len(pl) for pl in self.posting_lists], dtype=np.float64)
        self.sigma0 = float(sizes.std())
        self.eps0 = float(self._mean_recon_error())

        self.ada_stats = {
            "n_violations": 0,
            "n_local_reindex": 0,
            "n_global_rebuild": 0,
            "n_points_reclustered": 0,
        }

    # ------------------------------------------------------------------
    # similarity hook: inner product by default; overridden by L2 wrapper
    # ------------------------------------------------------------------
    def _similarity_matrix(self, A, B):
        """Pairwise similarity between rows of A and rows of B (inner product)."""
        return A @ B.T

    def _mean_recon_error(self):
        """Mean reconstruction error over non-empty partitions.

        eps_c = q_c/s_c - ||mu_c||^2 is the exact mean squared error of the
        partition's points around its centroid; for unit-normalized vectors
        q_c == s_c so it reduces to 1 - ||mu_c||^2.
        """
        sizes = np.array([len(pl) for pl in self.posting_lists[:self.k]], dtype=np.float64)
        nz = sizes > 0
        if not np.any(nz):
            return 0.0
        mu_norms = np.linalg.norm(self.centroids[:self.k], axis=1) ** 2
        eps = self.q_c[:self.k] / np.maximum(sizes, 1.0) - mu_norms
        return float(eps[nz].mean())

    def _find_cluster(self, vid):
        """O(1) PAM lookup (the paper uses PAM to support deletes)."""
        return self.pam.get(vid)

    def insert(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        if self.normalize:
            n = np.linalg.norm(vec)
            if n > 0.0:
                vec = vec / n
        return super().insert(vec)

    def _post_insert(self, c, i, vec):
        # Running-mean centroid update, incremental: the paper's
        # mu_new = mu + (s_delta/s_new)*(v - mu) with s_delta = +1 is
        # algebraically identical to maintaining the exact sum a_c and
        # deriving mu = a_c/s (no full recompute, no fp drift).
        self.a_c[c] += vec
        self.q_c[c] += float(np.dot(vec, vec))
        s_new = len(self.posting_lists[c])
        self.centroids[c] = self.a_c[c] / s_new
        self.pam[i] = c
        self._count_op()

    def _post_delete(self, c, i, vec):
        # s_delta = -1 in the paper's running-mean formula.
        self.a_c[c] -= vec
        self.q_c[c] -= float(np.dot(vec, vec))
        s_new = len(self.posting_lists[c])
        if s_new > 0:
            self.centroids[c] = self.a_c[c] / s_new
        else:
            self.centroids[c] = np.zeros(self.d, dtype=np.float32)
            self.a_c[c] = np.zeros(self.d, dtype=np.float32)
            self.q_c[c] = 0.0
        self.pam.pop(i, None)
        self._count_op()

    def _count_op(self):
        self.ops_since_check += 1
        if self.ops_since_check >= self.check_every:
            self.ops_since_check = 0
            self._run_checks()

    # ------------------------------------------------------------------
    # violation detection (vectorized over all partitions)
    # ------------------------------------------------------------------
    def _detect_violators(self):
        """f(c) = alpha*T*(beta*f_s + (1-beta)*f_d); returns violator indices.

        Two safeguards against pathological triggers from empty/tiny partitions:
          * size == 0 partitions never participate (the paper deletes empty
            partitions; under the slot layout we simply leave them inert).
          * the under-size denominator is clamped to max(s, tau_s/2.0) so f_s
            is bounded by 2.0. The paper's formula (tau_s - s)/max(s, 1.0) is
            unbounded (an empty slot scores tau_s, a 2-point micro-cluster
            scores tau_s/2) and makes micro partitions re-trigger every check;
            structural imbalance is instead handled by the balanced local
            reindex (_balance), not by unbounded scores.
        """
        k = self.k
        sizes = np.array([len(pl) for pl in self.posting_lists[:k]], dtype=np.float64)
        T = self.T[:k]
        tau_s = float(self.tau_s)
        nonempty = sizes > 0
        f_s = np.zeros(k, dtype=np.float64)
        # over-size: f_s = (s - tau_s)/tau_s (unbounded, as in the paper)
        over = nonempty & (sizes >= tau_s)
        f_s[over] = (sizes[over] - tau_s) / tau_s
        # under-size: denominator clamped to max(s, tau_s/2) -> f_s <= 2.0
        under = nonempty & ~over
        f_s[under] = (tau_s - sizes[under]) / np.maximum(sizes[under], tau_s / 2.0)
        mu = self.centroids[:k]
        mu0 = self.mu0[:k]
        mu0_norm = np.linalg.norm(mu0, axis=1)
        f_d = np.linalg.norm(mu - mu0, axis=1) / np.maximum(mu0_norm, 1e-8)
        f = self.alpha * T * (self.beta * f_s + (1.0 - self.beta) * f_d)
        return np.where(nonempty & (f > self.tau_f))[0]

    def _global_indicator(self):
        """G = gamma*|sigma-sigma0|/sigma0 + (1-gamma)*|eps-eps'|/eps'."""
        k = self.k
        sizes = np.array([len(pl) for pl in self.posting_lists[:k]], dtype=np.float64)
        sigma = float(sizes.std())
        eps = self._mean_recon_error()
        eps_prime = self.eps0 * (self.n0 / max(self.n, 1)) ** (1.0 / self.d)
        G = (self.gamma * abs(sigma - self.sigma0) / max(self.sigma0, 1e-8)
             + (1.0 - self.gamma) * abs(eps - eps_prime) / max(eps_prime, 1e-8))
        return G

    def _run_checks(self):
        violators = self._detect_violators()
        self.ada_stats["n_violations"] += int(len(violators))
        G = self._global_indicator()
        if G > self.tau_g:
            self._global_rebuild()
        elif len(violators) > 0:
            self._local_reindex(violators)

    # ------------------------------------------------------------------
    # global rebuild (safety valve)
    # ------------------------------------------------------------------
    def _global_rebuild(self):
        """Full k-means rebuild over all live vectors (G > tau_g fallback)."""
        t0 = time.perf_counter()
        live_ids = list(self.id_to_vec.keys())
        if len(live_ids) < self.k * 2:
            return
        vecs = np.array([self.id_to_vec[i] for i in live_ids], dtype=np.float32)
        centroids, assigns = kmeans_faiss(vecs, self.k, niter=25, seed=self.seed)
        self.centroids = centroids
        self.posting_lists = [[] for _ in range(self.k)]
        self.a_c = np.zeros((self.k, self.d), dtype=np.float32)
        self.q_c = np.zeros(self.k, dtype=np.float64)
        self.pam = {}
        for i, a in zip(live_ids, assigns):
            self.posting_lists[a].append(i)
            self.pam[i] = int(a)
        for c in range(self.k):
            pl = self.posting_lists[c]
            if pl:
                vecs_c = np.array([self.id_to_vec[i] for i in pl], dtype=np.float32)
                self.a_c[c] = vecs_c.sum(axis=0)
                self.q_c[c] = float(np.sum(vecs_c * vecs_c))
                self.centroids[c] = self.a_c[c] / len(pl)
            else:
                self.centroids[c] = np.zeros(self.d, dtype=np.float32)
        self.mu0 = self.centroids.copy()
        self.T = np.ones(self.k, dtype=np.float64)
        sizes = np.array([len(pl) for pl in self.posting_lists], dtype=np.float64)
        self.sigma0 = float(sizes.std())
        self.n0 = len(live_ids)
        self.eps0 = float(self._mean_recon_error())
        self.n_splits += 1
        self.ada_stats["n_global_rebuild"] += 1
        self.total_split_time += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # local reindex (Algorithm 4)
    # ------------------------------------------------------------------
    def _lloyd(self, X_vecs, init_centroids, n_iter):
        """Warm-started Lloyd iterations with inner-product assignment."""
        ctrs = init_centroids.astype(np.float32).copy()
        n = len(ctrs)
        for _ in range(n_iter):
            sims = self._similarity_matrix(X_vecs, ctrs)
            assigns = np.argmax(sims, axis=1)
            a_new = np.zeros_like(ctrs)
            np.add.at(a_new, assigns, X_vecs)
            counts = np.bincount(assigns, minlength=n).astype(np.float64)
            nz = counts > 0
            new_ctrs = np.zeros_like(ctrs)
            new_ctrs[nz] = a_new[nz] / counts[nz, None]
            new_ctrs[~nz] = ctrs[~nz]  # empty clusters keep previous centroid
            ctrs = new_ctrs
        assigns = np.argmax(self._similarity_matrix(X_vecs, ctrs), axis=1)
        return ctrs, assigns

    def _balance(self, assigns, X_vecs, ctrs, min_size):
        """Force every pool partition to >= min_size points (true balancing).

        Repeatedly takes points from the largest surplus partition and moves
        the ones most similar to an under-size partition's centroid into it
        (empty partitions are filled this way too, superseding the old
        _repair_empty which only stuffed 2 points into empties).  Donors are
        never drained below min_size, so every move strictly reduces the
        total deficit and the loop terminates.  If the pool holds fewer than
        min_size * n_partitions points the pool cannot sustain the floor and
        we return early (deficit stays; caller's centroid recompute is
        unaffected).  Affected centroids are recomputed by the caller from
        the returned assigns, per the existing write-back logic.
        """
        n_new = len(ctrs)
        counts = np.bincount(assigns, minlength=n_new).astype(np.int64)
        if counts.sum() < min_size * n_new:
            return assigns  # pool too small to balance every partition
        for recipient in np.where(counts < min_size)[0]:
            need = int(min_size - counts[recipient])
            while need > 0:
                surplus = counts - min_size
                donors = np.where(surplus > 0)[0]
                if len(donors) == 0:
                    break
                donor = int(donors[np.argmax(counts[donors])])
                give = min(need, int(counts[donor] - min_size))
                donor_pts = np.where(assigns == donor)[0]
                sims = self._similarity_matrix(X_vecs[donor_pts], ctrs[recipient:recipient + 1]).ravel()
                take = donor_pts[np.argsort(-sims)[:give]]
                assigns[take] = recipient
                counts[donor] -= give
                counts[recipient] += give
                need -= give
        return assigns

    def _local_reindex(self, violators):
        """Recluster the pool X around candidate centroids M (Algorithm 4)."""
        t0 = time.perf_counter()
        d = self.d
        violators = [int(c) for c in violators]
        released = set()
        M = []          # candidate centroids
        X = []          # point ids to recluster
        n_splits_this = 0

        # (a) violators: split oversized partitions, else centroid joins pool
        for c in violators:
            indices = [i for i in self.posting_lists[c] if i in self.id_to_vec]
            s = len(indices)
            if s == 0:
                # Fix 4: an empty violator contributes a zero centroid that can
                # attract no points, i.e. a guaranteed dead partition after the
                # pool Lloyd; skip it and leave the slot inert.
                continue
            released.add(c)
            if s > self.tau_s:
                k_c = max(2, min(int(np.ceil(s / self.tau_s)), s))
                vecs = np.array([self.id_to_vec[i] for i in indices], dtype=np.float32)
                ctrs, _ = kmeans_faiss(vecs, k_c, niter=10, seed=self.rng.randint(0, 2**31))
                for row in ctrs:
                    M.append(np.asarray(row, dtype=np.float32))
                n_splits_this += 1
            else:
                M.append(self.centroids[c].copy())
            X.extend(indices)

        # (b) KNN expansion: nearest r_c partitions to each pool centroid
        remaining = [c for c in range(self.k) if c not in released]
        if remaining:
            rem_ctrs = self.centroids[remaining]
            sims = self._similarity_matrix(np.asarray(M, dtype=np.float32), rem_ctrs)
            r = min(self.r_c, len(remaining))
            top = np.argsort(-sims, axis=1)[:, :r]
            neighbor_slots = set()
            for row in top:
                for ridx in row:
                    neighbor_slots.add(remaining[ridx])
            for c in neighbor_slots:
                indices = [i for i in self.posting_lists[c] if i in self.id_to_vec]
                X.extend(indices)
                M.append(self.centroids[c].copy())
                released.add(c)

        X = list(dict.fromkeys(X))  # dedupe (defensive; slots are disjoint)
        n_pool = len(X)
        min_size = max(2, int(0.75 * self.tau_s))

        # (c) pool over-allocation trim, warm-started Lloyd, balanced refine
        if n_pool > 0:
            X_vecs = np.array([self.id_to_vec[i] for i in X], dtype=np.float32)
            pool_ctrs = np.asarray(M, dtype=np.float32)
            # Fix 3: when the candidate pool has more centroids than the pool
            # points can sustain (n_pool // min_size), pre-scan assignments and
            # keep only the highest-mass candidates (pool-level merge: the
            # balanced k-means behavior of adapting the partition count to the
            # data). Dead centroids that attract no points are dropped too.
            max_candidates = max(1, n_pool // min_size)
            if len(pool_ctrs) > max_candidates:
                init_sims = self._similarity_matrix(X_vecs, pool_ctrs)
                init_assigns = np.argmax(init_sims, axis=1)
                mass = np.bincount(init_assigns, minlength=len(pool_ctrs))
                alive = np.where(mass > 0)[0]
                if len(alive) > max_candidates:
                    alive = alive[np.argsort(-mass[alive])[:max_candidates]]
                pool_ctrs = pool_ctrs[alive]
            ctrs, assigns = self._lloyd(X_vecs, pool_ctrs, self.iota)
            assigns = self._balance(assigns, X_vecs, ctrs, min_size)
            a_new = np.zeros((len(pool_ctrs), d), dtype=np.float32)
            np.add.at(a_new, assigns, X_vecs)
            counts = np.bincount(assigns, minlength=len(pool_ctrs)).astype(np.float64)
            ctrs_final = np.zeros((len(pool_ctrs), d), dtype=np.float32)
            nz = counts > 0
            ctrs_final[nz] = a_new[nz] / counts[nz, None]
            q_new = np.zeros(len(pool_ctrs), dtype=np.float64)
            np.add.at(q_new, assigns, np.sum(X_vecs * X_vecs, axis=1))
        else:
            X_vecs = np.zeros((0, d), dtype=np.float32)
            pool_ctrs = np.asarray(M, dtype=np.float32)
            ctrs_final = pool_ctrs.copy()
            a_new = ctrs_final.copy()
            q_new = np.zeros(len(pool_ctrs), dtype=np.float64)
            assigns = np.zeros(0, dtype=np.int64)

        # (e) write back into released slots (append when the pool grew)
        n_new = len(ctrs_final)
        slot_ids = sorted(released)
        while len(slot_ids) < n_new:
            slot_ids.append(self.k)
            self.posting_lists.append([])
            self.centroids = np.vstack([self.centroids, np.zeros((1, d), dtype=np.float32)])
            self.mu0 = np.vstack([self.mu0, np.zeros((1, d), dtype=np.float32)])
            self.a_c = np.vstack([self.a_c, np.zeros((1, d), dtype=np.float32)])
            self.q_c = np.append(self.q_c, 0.0)
            self.T = np.append(self.T, 1.0)
            self.k += 1

        old_cluster = np.array([self.pam.get(i, -1) for i in X], dtype=np.int64)
        for j, c in enumerate(slot_ids):
            if j < n_new:
                idxs = [X[t] for t in np.where(assigns == j)[0]] if n_pool > 0 else []
                self.posting_lists[c] = idxs
                self.centroids[c] = ctrs_final[j]
                self.mu0[c] = ctrs_final[j].copy()
                self.a_c[c] = a_new[j]
                self.q_c[c] = float(q_new[j])
                self.T[c] = 1.0
            else:
                self.posting_lists[c] = []
                self.centroids[c] = np.zeros(d, dtype=np.float32)
                self.mu0[c] = np.zeros(d, dtype=np.float32)
                self.a_c[c] = np.zeros(d, dtype=np.float32)
                self.q_c[c] = 0.0
                self.T[c] = 1.0

        # refresh PAM for pool points; count reassignments (SPFresh-aligned)
        if n_pool > 0:
            new_cluster = np.array([slot_ids[a] for a in assigns], dtype=np.int64)
            for i, nc in zip(X, new_cluster):
                self.pam[i] = int(nc)
            self.n_reassigns += int(np.sum(new_cluster != old_cluster))

        self.n_splits += n_splits_this
        self.ada_stats["n_local_reindex"] += 1
        self.ada_stats["n_points_reclustered"] += n_pool
        self.total_split_time += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # temperature updates driven by queries
    # ------------------------------------------------------------------
    def search(self, query, k=100, nprobe=8):
        """Search and update partition temperatures.

        With ``use_temperature=False`` the temperature update is skipped, so
        T stays 1.0 and detection degenerates to pure size+drift.
        """
        query = np.asarray(query, dtype=np.float32)
        if self.normalize:
            n = np.linalg.norm(query)
            if n > 0.0:
                query = query / n
        sims = np.dot(query, self.centroids[:self.k].T)
        candidates = np.argsort(-sims)[:min(nprobe, self.k)]
        result = super().search(query, k=k, nprobe=nprobe)
        if self.use_temperature:
            self._update_temperature(sims, candidates)
        return result

    def _update_temperature(self, sims, candidates):
        """Heat probed partitions, cool the rest (T floor 1.0)."""
        k = self.k
        if len(candidates) == 0:
            return
        sim_nearest = float(sims[candidates[0]])
        sel_sims = sims[candidates]
        ratio = np.ones(len(candidates), dtype=np.float64)
        pos = sel_sims > 0.0
        ratio[pos] = sim_nearest / sel_sims[pos]
        factor = np.clip(1.0 + self.eta * ratio, 1.0, 2.0)
        self.T[candidates] *= factor
        unsel = np.ones(k, dtype=bool)
        unsel[candidates] = False
        self.T[unsel] = np.maximum(self.T[unsel] * (1.0 - self.nu), 1.0)


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
    elif name in ("adaivf",):
        return AdaIVF(data, nlist, **kwargs)
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
