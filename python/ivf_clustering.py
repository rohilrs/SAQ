"""
Clustering script for IVF index construction.

This script performs K-means clustering on vector data using FAISS,
producing centroid vectors and cluster assignments needed for
IVF index construction in SAQ.

Usage:
    python ivf_clustering.py <data_file> <num_clusters> <centroids_out> <cluster_ids_out> [metric]

Arguments:
    data_file:       Path to input vectors (.fvecs or .npy format)
    num_clusters:    Number of clusters (K). Recommend: 4 * sqrt(n_vectors)
    centroids_out:   Output path for centroid vectors (.fvecs or .npy)
    cluster_ids_out: Output path for cluster assignments (.ivecs or .npy)
    metric:          Distance metric: 'l2' (default) or 'ip' (inner product)

Example:
    python ivf_clustering.py data/sift_base.fvecs 4096 data/sift_centroids.fvecs data/sift_cluster_ids.ivecs l2

Requirements:
    - faiss-cpu or faiss-gpu
    - numpy
"""

import argparse
import os
import struct
import sys
import time
from typing import Tuple

import numpy as np

try:
    import faiss
except ImportError:
    print("Error: faiss not installed. Install with:")
    print("  pip install faiss-cpu")
    print("  or")
    print("  pip install faiss-gpu")
    sys.exit(1)


def read_fvecs(filename: str) -> np.ndarray:
    """Read vectors from .fvecs format."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def write_fvecs(filename: str, vectors: np.ndarray):
    """Write vectors to .fvecs format."""
    with open(filename, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack('i', dim))
            f.write(vec.astype(np.float32).tobytes())


def write_ivecs(filename: str, indices: np.ndarray):
    """Write indices to .ivecs format."""
    with open(filename, 'wb') as f:
        for idx in indices:
            # Each row is a single index for cluster assignment
            if np.isscalar(idx):
                f.write(struct.pack('i', 1))
                f.write(struct.pack('i', int(idx)))
            else:
                dim = len(idx)
                f.write(struct.pack('i', dim))
                f.write(idx.astype(np.int32).tobytes())


def load_vectors(filename: str) -> np.ndarray:
    """Load vectors from file (supports .fvecs and .npy)."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.fvecs':
        return read_fvecs(filename)
    elif ext == '.npy':
        return np.load(filename).astype(np.float32)
    elif ext == '.bin':
        # Assume raw float32 format with first 8 bytes as (n, dim)
        with open(filename, 'rb') as f:
            n, dim = struct.unpack('II', f.read(8))
            data = np.frombuffer(f.read(n * dim * 4), dtype=np.float32)
            return data.reshape(n, dim)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_centroids(filename: str, centroids: np.ndarray):
    """Save centroids to file (supports .fvecs and .npy)."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.fvecs':
        write_fvecs(filename, centroids)
    elif ext == '.npy':
        np.save(filename, centroids)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_cluster_ids(filename: str, cluster_ids: np.ndarray):
    """Save cluster IDs to file (supports .ivecs and .npy)."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.ivecs':
        write_ivecs(filename, cluster_ids.flatten())
    elif ext == '.npy':
        np.save(filename, cluster_ids.astype(np.int32))
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def perform_clustering(
    data: np.ndarray,
    num_clusters: int,
    metric: str = 'l2',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering using FAISS.
    
    Args:
        data: Input vectors (n x dim)
        num_clusters: Number of clusters (K)
        metric: 'l2' for Euclidean, 'ip' for inner product
        verbose: Print progress
        
    Returns:
        Tuple of (centroids, cluster_ids)
    """
    n, dim = data.shape
    
    # Choose metric
    if metric.lower() in ['l2', 'euclidean']:
        faiss_metric = faiss.METRIC_L2
        metric_name = "L2"
    elif metric.lower() in ['ip', 'innerproduct', 'inner_product']:
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        metric_name = "InnerProduct"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if verbose:
        print(f"Clustering {n:,} vectors of dimension {dim}")
        print(f"Target clusters: {num_clusters}")
        print(f"Distance metric: {metric_name}")
    
    # Create IVF index for clustering
    index_str = f"IVF{num_clusters},Flat"
    index = faiss.index_factory(dim, index_str, faiss_metric)
    index.verbose = verbose
    
    # Train the index (this runs K-means)
    t0 = time.time()
    index.train(data)
    train_time = time.time() - t0
    
    if verbose:
        print(f"Clustering completed in {train_time:.2f} seconds")
    
    # Extract centroids from the quantizer
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    
    # Assign each vector to its nearest cluster
    if verbose:
        print("Computing cluster assignments...")
    
    t0 = time.time()
    _, cluster_ids = index.quantizer.search(data, 1)
    assign_time = time.time() - t0
    
    if verbose:
        print(f"Assignment completed in {assign_time:.2f} seconds")
    
    # Compute cluster statistics
    cluster_sizes = np.bincount(cluster_ids.flatten(), minlength=num_clusters)
    
    if verbose:
        print(f"\nCluster statistics:")
        print(f"  Min size:  {cluster_sizes.min()}")
        print(f"  Max size:  {cluster_sizes.max()}")
        print(f"  Avg size:  {cluster_sizes.mean():.1f}")
        print(f"  Std size:  {cluster_sizes.std():.1f}")
        empty_clusters = np.sum(cluster_sizes == 0)
        if empty_clusters > 0:
            print(f"  Warning: {empty_clusters} empty cluster(s)")
    
    return centroids.astype(np.float32), cluster_ids.flatten().astype(np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering for IVF index construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python ivf_clustering.py data/sift_base.fvecs 4096 \\
        data/sift_centroids.fvecs data/sift_cluster_ids.ivecs l2
        
Supported file formats:
    Input:  .fvecs, .npy, .bin
    Output: .fvecs/.npy (centroids), .ivecs/.npy (cluster IDs)
        """
    )
    
    parser.add_argument('data_file', help='Input vector file')
    parser.add_argument('num_clusters', type=int, help='Number of clusters (K)')
    parser.add_argument('centroids_out', help='Output centroids file')
    parser.add_argument('cluster_ids_out', help='Output cluster IDs file')
    parser.add_argument('metric', nargs='?', default='l2',
                        choices=['l2', 'ip'], help='Distance metric (default: l2)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Load data
    verbose = not args.quiet
    if verbose:
        print(f"Loading vectors from {args.data_file}...")
    
    data = load_vectors(args.data_file)
    n, dim = data.shape
    
    if verbose:
        print(f"Loaded {n:,} vectors of dimension {dim}")
    
    # Validate num_clusters
    if args.num_clusters <= 0:
        print("Error: num_clusters must be positive")
        sys.exit(1)
    
    if args.num_clusters > n:
        print(f"Warning: num_clusters ({args.num_clusters}) > n_vectors ({n})")
        print(f"Reducing to {n}")
        args.num_clusters = n
    
    # Perform clustering
    centroids, cluster_ids = perform_clustering(
        data, args.num_clusters, args.metric, verbose
    )
    
    # Save results
    if verbose:
        print(f"\nSaving centroids to {args.centroids_out}...")
    save_centroids(args.centroids_out, centroids)
    
    if verbose:
        print(f"Saving cluster IDs to {args.cluster_ids_out}...")
    save_cluster_ids(args.cluster_ids_out, cluster_ids)
    
    if verbose:
        print("\nDone!")


if __name__ == "__main__":
    main()
