"""
K-means clustering via Faiss for IVF index construction.

Clusters base vectors into K partitions, producing centroid vectors and
per-vector cluster assignments.

Ported from reference SAQ repository (python/ivf.py).

Usage:
    python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k --num-clusters 4096
    python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k --num-clusters 4096 --metric ip

Data layout expected:
    <data_dir>/vectors.fvecs      -- base vectors (or specify --base-file)

Output:
    <output_dir>/centroids_<K>.fvecs
    <output_dir>/cluster_ids_<K>.ivecs
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    print("Error: faiss not installed. Install with: pip install faiss-cpu")
    sys.exit(1)

from preprocessing.utils.io import read_somefiles, write_fvecs, write_ivecs


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering via Faiss for IVF index construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k --num-clusters 4096
    python -m preprocessing.ivf --data-dir data/datasets/dbpedia_100k -K 4096 --metric ip
        """,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing base vectors",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as data-dir)",
    )
    parser.add_argument(
        "--base-file",
        default="vectors.fvecs",
        help="Base vectors filename (default: vectors.fvecs)",
    )
    parser.add_argument(
        "--num-clusters",
        "-K",
        type=int,
        default=4096,
        help="Number of clusters (default: 4096)",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "ip"],
        default="l2",
        help="Distance metric: l2 (default) or ip (inner product)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir
    K = args.num_clusters
    metric = args.metric.upper()

    os.makedirs(output_dir, exist_ok=True)

    # Load base vectors
    data_path = os.path.join(data_dir, args.base_file)
    print(f"Clustering - {data_path} with {K} clusters using {metric} distance")
    X = read_somefiles(data_path)

    D = X.shape[1]
    print(f"Data shape: {X.shape}")

    # Output file suffixes
    suffix = ".ip" if metric == "IP" else ""
    centroids_path = os.path.join(output_dir, f"centroids_{K}{suffix}.fvecs")
    cluster_id_path = os.path.join(output_dir, f"cluster_ids_{K}{suffix}.ivecs")

    # Create and train the IVF index (runs K-means internally)
    if metric == "IP":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, K)
    else:
        index = faiss.index_factory(D, f"IVF{K},Flat")

    index.verbose = True

    start_time = time.time()
    index.train(X)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Extract centroids
    centroids = index.quantizer.reconstruct_n(0, index.nlist)

    # Compute cluster assignments
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)

    # For L2, Faiss returns squared distances; take square root
    if metric == "L2":
        dist_to_centroid = dist_to_centroid ** 0.5

    # Save results
    write_ivecs(cluster_id_path, cluster_id)
    write_fvecs(centroids_path, centroids)

    print(f"Centroids saved to {centroids_path}")
    print(f"Cluster IDs saved to {cluster_id_path}")

    # Print cluster statistics
    cluster_sizes = np.bincount(cluster_id.flatten(), minlength=K)
    print(f"\nCluster statistics:")
    print(f"  Min size:  {cluster_sizes.min()}")
    print(f"  Max size:  {cluster_sizes.max()}")
    print(f"  Avg size:  {cluster_sizes.mean():.1f}")
    print(f"  Std size:  {cluster_sizes.std():.1f}")
    empty_clusters = np.sum(cluster_sizes == 0)
    if empty_clusters > 0:
        print(f"  Warning: {empty_clusters} empty cluster(s)")

    print("Done!")


if __name__ == "__main__":
    main()
