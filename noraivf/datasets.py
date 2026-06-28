"""
NovaIVF Experiment Suite — Datasets
Generates synthetic data mimicking embodied AI vector search scenarios:
spatial locality, distribution drift, and cluster imbalance.
"""
import numpy as np
import os


def compute_l2_ground_truth(train, queries, k=100, batch_size=256):
    """Compute exact Euclidean nearest neighbors for query vectors."""
    k = min(k, len(train))
    gt = np.zeros((len(queries), k), dtype=np.int32)
    train_sq = np.sum(train.astype(np.float32) ** 2, axis=1)
    for i in range(0, len(queries), batch_size):
        end = min(i + batch_size, len(queries))
        q = queries[i:end].astype(np.float32)
        q_sq = np.sum(q ** 2, axis=1, keepdims=True)
        dists = q_sq + train_sq[None, :] - 2.0 * np.dot(q, train.T)
        gt[i:end] = np.argsort(dists, axis=1)[:, :k]
    return gt


def load_ann_benchmark_hdf5(path, max_train=None, max_queries=None, k=100):
    """
    Load an ann-benchmarks Euclidean HDF5 dataset.

    Expected datasets are `train`, `test`, and optionally `neighbors`.
    If `neighbors` is not present, exact L2 ground truth is computed.
    """
    import h5py

    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"][:], dtype=np.float32)
        queries = np.asarray(f["test"][:], dtype=np.float32)
        if max_train is not None:
            train = train[:max_train]
        if max_queries is not None:
            queries = queries[:max_queries]

        if "neighbors" in f:
            gt = np.asarray(f["neighbors"][:], dtype=np.int32)
            if max_queries is not None:
                gt = gt[:max_queries]
            gt = gt[:, : min(k, gt.shape[1])]
            if max_train is not None:
                valid = gt < max_train
                if not np.all(valid):
                    gt = compute_l2_ground_truth(train, queries, k=k)
        else:
            gt = compute_l2_ground_truth(train, queries, k=k)

    return train, queries, gt


def load_npy_vector_dataset_l2(base_dir, max_train=None, max_queries=None, k=100,
                               recompute_gt=False):
    """
    Load a vector dataset stored as train.npy, queries.npy, and optional gt.npy.

    If `gt.npy` is missing, incompatible with a max_train subset, or
    `recompute_gt=True`, exact Euclidean ground truth is computed.
    """
    train = np.asarray(np.load(os.path.join(base_dir, "train.npy")), dtype=np.float32)
    queries = np.asarray(np.load(os.path.join(base_dir, "queries.npy")), dtype=np.float32)
    if max_train is not None:
        train = train[:max_train]
    if max_queries is not None:
        queries = queries[:max_queries]

    gt_path = os.path.join(base_dir, "gt.npy")
    if os.path.exists(gt_path) and not recompute_gt:
        gt = np.asarray(np.load(gt_path), dtype=np.int32)
        if max_queries is not None:
            gt = gt[:max_queries]
        gt = gt[:, : min(k, gt.shape[1])]
        if max_train is not None and not np.all(gt < max_train):
            gt = compute_l2_ground_truth(train, queries, k=k)
    else:
        gt = compute_l2_ground_truth(train, queries, k=k)

    return train, queries, gt


def generate_embodied_dataset(
    n_vectors=50_000,
    n_queries=5_000,
    d=512,
    n_clusters=50,
    cluster_std=0.35,
    drift_frac=0.3,
    seed=42,
):
    """
    Generate a dataset mimicking embodied AI streaming vector data.

    Characteristics:
    - Multiple Gaussian clusters (spatial locality → natural NPA boundary cases)
    - Variable cluster sizes (imbalance → tests partition balance strategies)
    - Temporal drift: a subset of clusters shift over time (distribution drift)

    Returns train (n_vectors, d), test (n_queries, d), ground_truth (n_queries, 100)
    """
    rng = np.random.RandomState(seed)

    # Cluster sizes: log-normal to create imbalance
    log_weights = rng.randn(n_clusters) * 0.8
    weights = np.exp(log_weights)
    weights /= weights.sum()
    cluster_sizes = (weights * n_vectors).astype(int)
    cluster_sizes[-1] += n_vectors - cluster_sizes.sum()

    # Generate cluster centers
    centers = rng.randn(n_clusters, d) * 2.0

    # Apply drift to some clusters (simulating temporal change)
    n_drift = int(n_clusters * drift_frac)
    drift_clusters = rng.choice(n_clusters, n_drift, replace=False)
    drift_direction = rng.randn(n_drift, d)
    drift_direction /= np.linalg.norm(drift_direction, axis=1, keepdims=True)

    train = np.zeros((n_vectors, d), dtype=np.float32)
    labels = np.zeros(n_vectors, dtype=np.int32)
    offset = 0
    drift_scale = 0

    for c in range(n_clusters):
        size = cluster_sizes[c]
        if size <= 0:
            continue
        # Half the points near original center, half drifted
        half = size // 2
        if c in drift_clusters:
            idx = np.where(drift_clusters == c)[0][0]
            drift_scale = rng.uniform(0.5, 2.0)
        else:
            drift_scale = 0

        # Original cluster points
        pts1 = centers[c] + rng.randn(half, d) * cluster_std
        train[offset : offset + half] = pts1.astype(np.float32)
        labels[offset : offset + half] = c

        # Drifted points (if applicable)
        pts2 = (centers[c] + drift_direction[drift_clusters == c][0] * drift_scale
                if c in drift_clusters else centers[c]) + rng.randn(size - half, d) * cluster_std
        train[offset + half : offset + size] = pts2.astype(np.float32)
        labels[offset + half : offset + size] = c
        offset += size

    train = train[:offset]
    labels = labels[:offset]
    n_vectors = offset

    # Shuffle
    perm = rng.permutation(n_vectors)
    train = train[perm]
    labels = labels[perm]

    # Generate queries from the same distribution
    query_centers = centers + rng.randn(*centers.shape) * 0.2
    queries = np.zeros((n_queries, d), dtype=np.float32)
    for i in range(n_queries):
        c = rng.choice(n_clusters)
        queries[i] = (query_centers[c] + rng.randn(d) * cluster_std * 0.8).astype(np.float32)

    # Normalize to unit norm (cosine similarity → L2 equivalence)
    train_norms = np.linalg.norm(train, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1
    train = train / train_norms
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms[query_norms == 0] = 1
    queries = queries / query_norms

    # Compute ground truth (top-100 nearest neighbors) via brute force
    print("Computing ground truth (brute-force KNN)...")
    gt = np.zeros((n_queries, 100), dtype=np.int32)
    batch_size = 500
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        dists = np.dot(queries[i:end], train.T)
        gt[i:end] = np.argsort(-dists, axis=1)[:, :100]
    print(f"Dataset generated: {n_vectors} train vectors, {n_queries} queries, d={d}")

    return train, queries, gt


def compute_centroids(data, n_clusters=50, n_iter=20, seed=42):
    """
    Run k-means on data to get cluster centroids.
    Returns (centroids, cluster_assignments).
    """
    rng = np.random.RandomState(seed)
    n, d = data.shape
    
    # Initialize centroids via k-means++
    idx = rng.choice(n, 1)
    centroids = np.zeros((n_clusters, d), dtype=np.float32)
    centroids[0] = data[idx[0]]
    
    # Distances to nearest centroid for each point
    min_dists = np.full(n, np.inf)
    
    for c in range(1, n_clusters):
        # Update distances
        new_dists = np.sum((data - centroids[c-1]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        # Sample next centroid with probability proportional to squared distance
        probs = min_dists / min_dists.sum()
        chosen = rng.choice(n, p=probs)
        centroids[c] = data[chosen]
    
    # Lloyd iteration
    for it in range(n_iter):
        # Assign points to nearest centroid
        dists = np.dot(data, centroids.T)  # cosine similarity (data is normalized)
        # For normalized vectors, max dot = min L2
        assignments = np.argmax(dists, axis=1)
        
        # Update centroids
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = data[mask].mean(axis=0)
        
        # Re-normalize
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids /= norms
    
    # Final assignment
    dists = np.dot(data, centroids.T)
    assignments = np.argmax(dists, axis=1)
    
    return centroids.astype(np.float32), assignments


def generate_boundary_queries(data, centroids, n_queries=5000, 
                               boundary_threshold=0.85, pool_multiplier=10, seed=42):
    """
    Generate query points that lie near partition boundaries.
    
    A point is "on the boundary" when its distance to the nearest centroid 
    is close to its distance to the second-nearest centroid. We define:
    
        boundary_score = cos_dist_to_nearest / cos_dist_to_second_nearest
    
    where higher score = closer to boundary (maximum 1.0 when equidistant).
    
    Args:
        data: training data (N, d), used to compute ground truth
        centroids: cluster centroids (n_clusters, d)
        n_queries: number of boundary queries to generate
        boundary_threshold: minimum boundary score to keep (0.0 - 1.0)
        pool_multiplier: generate pool_size = n_queries * pool_multiplier candidates
        seed: random seed
    
    Returns:
        boundary_queries: (n_queries, d) boundary query vectors
        boundary_gt: (n_queries, 100) ground truth top-100 neighbors
    """
    rng = np.random.RandomState(seed)
    n, d = data.shape
    n_clusters = len(centroids)
    
    # Generate a large candidate pool: sample points BETWEEN pairs of nearby centroids
    pool_size = n_queries * pool_multiplier
    candidates = np.zeros((pool_size, d), dtype=np.float32)
    candidate_scores = np.zeros(pool_size, dtype=np.float32)
    
    # Strategy: generate points at midpoints between random centroid pairs,
    # with noise to create a natural boundary distribution
    for i in range(pool_size):
        # Pick two random centroids
        c1, c2 = rng.choice(n_clusters, 2, replace=False)
        
        # Interpolate between them with random weight near 0.5 (the decision boundary)
        t = rng.uniform(0.35, 0.65)  # weight near 0.5 = boundary region
        midpoint = centroids[c1] * t + centroids[c2] * (1 - t)
        
        # Add noise perpendicular to the boundary direction
        noise = rng.randn(d).astype(np.float32) * 0.15
        candidates[i] = midpoint + noise
        
        # Normalize
        norm = np.linalg.norm(candidates[i])
        if norm > 0:
            candidates[i] /= norm
    
    # Compute boundary scores
    sims = np.dot(candidates, centroids.T)  # (pool_size, n_clusters) cosine similarities
    
    # Get top-2 similarities, compute boundary score
    sorted_sims = -np.sort(-sims, axis=1)  # negate for descending sort
    nearest = sorted_sims[:, 0]
    second_nearest = sorted_sims[:, 1]
    
    # boundary_score = sim_to_second / sim_to_first → 1.0 means on boundary
    # Use arccos distances for better separation
    eps = 1e-8
    d1 = np.arccos(np.clip(nearest, -1 + eps, 1 - eps))
    d2 = np.arccos(np.clip(second_nearest, -1 + eps, 1 - eps))
    
    # Score: d1/d2 → when d1 ≈ d2, score ≈ 1 (boundary)
    boundary_scores = d1 / (d2 + eps)
    
    # Select queries with highest boundary scores
    idx = np.argsort(-boundary_scores)  # descending
    kept = 0
    selected = []
    for i in idx:
        if boundary_scores[i] >= boundary_threshold:
            selected.append(i)
            kept += 1
            if kept >= n_queries:
                break
    
    if kept < n_queries:
        # If not enough above threshold, take the top n_queries
        selected = idx[:n_queries]
    
    boundary_queries = candidates[selected].astype(np.float32)
    boundary_scores = boundary_scores[selected]
    
    print(f"Boundary queries selected: {len(boundary_queries)}/{pool_size}")
    print(f"  Boundary score range: [{boundary_scores.min():.4f}, {boundary_scores.max():.4f}]")
    print(f"  Threshold: {boundary_threshold}")
    
    # Compute ground truth for boundary queries
    print("Computing ground truth for boundary queries (brute-force KNN)...")
    boundary_gt = np.zeros((len(boundary_queries), 100), dtype=np.int32)
    batch_size = 500
    for i in range(0, len(boundary_queries), batch_size):
        end = min(i + batch_size, len(boundary_queries))
        dists = np.dot(boundary_queries[i:end], data.T)
        boundary_gt[i:end] = np.argsort(-dists, axis=1)[:, :100]
    
    return boundary_queries, boundary_gt


if __name__ == "__main__":
    train, queries, gt = generate_embodied_dataset()
    print(f"Train shape: {train.shape}")
    print(f"Queries shape: {queries.shape}")
    print(f"GT shape: {gt.shape}")
    
    # Demo: generate boundary queries
    print("\n--- Boundary Query Demo ---")
    centroids, _ = compute_centroids(train, n_clusters=50)
    bq, bgt = generate_boundary_queries(train, centroids, n_queries=1000)
    print(f"Boundary queries shape: {bq.shape}")
    print(f"Boundary GT shape: {bgt.shape}")
