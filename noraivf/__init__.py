"""
NoraIVF: A memory-native dynamic IVF index for streaming vector search.

Implements the SCDKM (Streaming Coordinate Descent K-Means) protocol for
split operations on IVF posting lists without explicit boundary repair.
"""

from noraivf.core import (
    kmeans_faiss,
    brute_force_knn,
    IVFIndex,
    FrozenIVF,
    RebuildIVF,
    UpdateCentroidsIVF,
    DeDriftIVF,
    SPFreshIVF,
    NovaIVF,
    create_index,
)

from noraivf.datasets import (
    compute_l2_ground_truth,
    load_ann_benchmark_hdf5,
    load_npy_vector_dataset_l2,
    generate_embodied_dataset,
    compute_centroids,
    generate_boundary_queries,
)

__all__ = [
    "kmeans_faiss",
    "brute_force_knn",
    "IVFIndex",
    "FrozenIVF",
    "RebuildIVF",
    "UpdateCentroidsIVF",
    "DeDriftIVF",
    "SPFreshIVF",
    "NovaIVF",
    "create_index",
    "compute_l2_ground_truth",
    "load_ann_benchmark_hdf5",
    "load_npy_vector_dataset_l2",
    "generate_embodied_dataset",
    "compute_centroids",
    "generate_boundary_queries",
]
