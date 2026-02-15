"""
Brute-force ground truth computation for nearest neighbor evaluation.

Computes the exact K nearest neighbors for each query vector against
the base vectors using L2 distance, with multi-threaded parallelism.

Ported from reference SAQ repository (python/compute_gt.py).

Usage:
    python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k
    python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k --top-k 100 --threads 16

Data layout expected:
    <data_dir>/vectors.fvecs     -- base vectors
    <data_dir>/queries.fvecs     -- query vectors

Output:
    <output_dir>/groundtruth.ivecs
"""

import argparse
import concurrent.futures
import os
import sys

import numpy as np

from preprocessing.utils.io import read_fvecs, write_ivecs


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force ground truth computation (L2 distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k
    python -m preprocessing.compute_gt --data-dir data/datasets/dbpedia_100k --top-k 100
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
        "--output-file",
        default="groundtruth.ivecs",
        help="Output filename (default: groundtruth.ivecs)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of nearest neighbors to compute (default: 1000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for parallel computation (default: 8)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir
    top_k = args.top_k
    num_threads = args.threads

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    base_path = os.path.join(data_dir, args.base_file)
    query_path = os.path.join(data_dir, args.query_file)

    print(f"Loading base vectors from {base_path}...")
    base = read_fvecs(base_path)
    print(f"Base shape: {base.shape}")

    print(f"Loading query vectors from {query_path}...")
    query = read_fvecs(query_path)
    print(f"Query shape: {query.shape}")

    print(
        f"Computing ground truth for {len(query)} queries "
        f"(top-{top_k}) using {num_threads} threads..."
    )

    def process_query(q):
        """Process a single query vector and return its top-K nearest neighbors."""
        distances = np.linalg.norm(base - q, axis=1)
        return list(np.argsort(distances))[:top_k]

    # Process queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        gt = list(executor.map(process_query, [q for q in query]))

    gt = np.array(gt)

    output_path = os.path.join(output_dir, args.output_file)
    print(f"Writing results to {output_path}")
    write_ivecs(output_path, gt)
    print("Done!")


if __name__ == "__main__":
    main()
