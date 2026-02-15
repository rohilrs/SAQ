"""
PCA preprocessing via Faiss PCAMatrix.

Trains PCA on base vectors, then applies the transformation to base, query,
and (optionally) centroid vectors. Saves the PCA mean, transformation matrix,
transformed vectors, and per-dimension variances.

Ported from reference SAQ repository (python/pca.py).

Usage:
    python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k
    python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k --pca-dim 256
    python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k --centroids centroids_4096.fvecs

Data layout expected:
    <data_dir>/vectors.fvecs      -- base vectors
    <data_dir>/queries.fvecs      -- query vectors
    <data_dir>/<centroids>        -- centroid vectors (optional)

Output:
    <output_dir>/pca_mean.fvecs
    <output_dir>/pca_matrix.fvecs
    <output_dir>/vectors_pca.fvecs
    <output_dir>/queries_pca.fvecs
    <output_dir>/centroids_pca.fvecs  (if centroids provided)
    <output_dir>/variances_pca.fvecs
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

from preprocessing.utils.io import read_fvecs, write_fvecs


def main():
    parser = argparse.ArgumentParser(
        description="PCA preprocessing via Faiss PCAMatrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k
    python -m preprocessing.pca --data-dir data/datasets/dbpedia_100k --pca-dim 256
        """,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing vectors.fvecs and queries.fvecs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as data-dir)",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="PCA output dimension (default: 0 = same as input, full rotation)",
    )
    parser.add_argument(
        "--base-file",
        default="vectors.fvecs",
        help="Base vectors filename (default: vectors.fvecs)",
    )
    parser.add_argument(
        "--query-file",
        default="queries.fvecs",
        help="Query vectors filename (default: queries.fvecs)",
    )
    parser.add_argument(
        "--centroids",
        default=None,
        help="Centroid vectors filename (optional, e.g. centroids_4096.fvecs)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only process base vectors (skip queries and centroids)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load base vectors
    base_path = os.path.join(data_dir, args.base_file)
    print(f"Loading base vectors from {base_path}...")
    X_base = read_fvecs(base_path)
    print(f"Base data shape: {X_base.shape}")

    D_IN = X_base.shape[1]
    D_OUT = args.pca_dim if args.pca_dim > 0 else D_IN

    # Load query and centroid vectors if needed
    X_query = None
    X_centroid = None

    if not args.base_only:
        query_path = os.path.join(data_dir, args.query_file)
        if os.path.exists(query_path):
            X_query = read_fvecs(query_path)
            print(f"Query data shape: {X_query.shape}")
        else:
            print(f"Warning: query file not found at {query_path}, skipping")

        if args.centroids:
            centroid_path = os.path.join(data_dir, args.centroids)
            if os.path.exists(centroid_path):
                X_centroid = read_fvecs(centroid_path)
                print(f"Centroid data shape: {X_centroid.shape}")
            else:
                print(f"Warning: centroid file not found at {centroid_path}, skipping")

    # Initialize and train PCA
    print(f"Training PCA (D_IN={D_IN}, D_OUT={D_OUT})...")
    pca_matrix = faiss.PCAMatrix(D_IN, D_OUT)
    start_time = time.time()
    pca_matrix.train(X_base)
    print(f"Training time: {time.time() - start_time:.3f} seconds")

    # Extract PCA components
    print("Extracting PCA components...")
    eigen_vecs = faiss.vector_to_array(pca_matrix.eigenvalues)
    pca_matrix_array = faiss.vector_to_array(pca_matrix.PCAMat).reshape(D_OUT, D_IN)
    pca_mean = faiss.vector_to_array(pca_matrix.mean).reshape(1, D_IN)

    print(f"Eigenvalues (first 10): {eigen_vecs[:10]}")

    # Apply PCA transformation
    print("Applying PCA transformation...")
    start_time = time.time()
    X_base_transformed = pca_matrix.apply(X_base)

    if X_query is not None:
        X_query_transformed = pca_matrix.apply(X_query)
    if X_centroid is not None:
        X_centroid_transformed = pca_matrix.apply(X_centroid)

    print(f"Transformation time: {time.time() - start_time:.3f} seconds")

    # Save PCA components
    pca_mean_path = os.path.join(output_dir, "pca_mean.fvecs")
    pca_mat_path = os.path.join(output_dir, "pca_matrix.fvecs")
    write_fvecs(pca_mean_path, pca_mean)
    write_fvecs(pca_mat_path, pca_matrix_array)
    print(f"PCA mean saved to {pca_mean_path}")
    print(f"PCA matrix saved to {pca_mat_path}")

    # Save transformed vectors
    transformed_base_path = os.path.join(output_dir, "vectors_pca.fvecs")
    write_fvecs(transformed_base_path, X_base_transformed)
    print(f"Transformed base data saved to {transformed_base_path}")

    if X_query is not None:
        transformed_query_path = os.path.join(output_dir, "queries_pca.fvecs")
        write_fvecs(transformed_query_path, X_query_transformed)
        print(f"Transformed query data saved to {transformed_query_path}")

    if X_centroid is not None:
        transformed_centroid_path = os.path.join(output_dir, "centroids_pca.fvecs")
        write_fvecs(transformed_centroid_path, X_centroid_transformed)
        print(f"Transformed centroid data saved to {transformed_centroid_path}")

    # Compute and save per-dimension variances of transformed data
    base_variance = np.var(X_base_transformed, axis=0).reshape(1, -1)
    variance_path = os.path.join(output_dir, "variances_pca.fvecs")
    write_fvecs(variance_path, base_variance)
    print(f"Per-dimension variances saved to {variance_path}")
    print(f"Variance shape: {base_variance.shape}")
    print(f"Variances (first 10): {base_variance[0, :10]}")

    print("PCA completed.")


if __name__ == "__main__":
    main()
