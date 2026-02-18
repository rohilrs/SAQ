"""Benchmark script for SAQ Python bindings.

Mirrors the C++ IVF benchmark for performance comparison.
"""

import argparse
import sys
import time

import numpy as np

try:
    import saq
except ImportError:
    # Try adding parent dir to path for development builds
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import saq


def simple_kmeans(data: np.ndarray, k: int, n_iter: int = 10,
                  seed: int = 42) -> tuple:
    """Simple k-means clustering (for benchmarking without faiss)."""
    rng = np.random.RandomState(seed)
    n, dim = data.shape

    # Random initialization
    indices = rng.choice(n, k, replace=False)
    centroids = data[indices].copy()

    assignments = np.zeros(n, dtype=np.uint32)

    for _ in range(n_iter):
        # Assign vectors to nearest centroid
        # Compute distances: ||x - c||^2 = ||x||^2 - 2*x*c^T + ||c||^2
        x_sq = np.sum(data ** 2, axis=1, keepdims=True)   # (n, 1)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, k)
        dists = x_sq - 2 * data @ centroids.T + c_sq  # (n, k)
        assignments = np.argmin(dists, axis=1).astype(np.uint32)

        # Update centroids
        for c in range(k):
            mask = assignments == c
            if np.any(mask):
                centroids[c] = data[mask].mean(axis=0)

    return centroids, assignments


def compute_ground_truth(data: np.ndarray, queries: np.ndarray,
                         k: int) -> np.ndarray:
    """Brute-force k-NN ground truth."""
    nq = queries.shape[0]
    gt = np.zeros((nq, k), dtype=np.uint32)

    for q in range(nq):
        dists = np.sum((data - queries[q]) ** 2, axis=1)
        gt[q] = np.argpartition(dists, k)[:k]
        # Sort the top-k
        gt[q] = gt[q][np.argsort(dists[gt[q]])]

    return gt


def compute_recall(results_indices: np.ndarray,
                   ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    nq = results_indices.shape[0]
    hits = 0
    for q in range(nq):
        gt_set = set(ground_truth[q, :k].tolist())
        for i in range(min(k, results_indices.shape[1])):
            if results_indices[q, i] in gt_set:
                hits += 1
    return hits / (nq * k)


def run_benchmark(n_vectors: int, dim: int, n_clusters: int,
                  n_queries: int, k: int, nprobe: int,
                  bpd: float, seed: int = 42):
    """Run a single benchmark configuration."""
    print(f"\n=== IVF Scalar Search Benchmark (Python) ===")
    print(f"Vectors: {n_vectors}, Dim: {dim}, Clusters: {n_clusters}")
    print(f"Queries: {n_queries}, k: {k}, nprobe: {nprobe}")
    print(f"Bits per dim: {bpd}")
    print()

    rng = np.random.RandomState(seed)

    # Generate data
    print("Generating data...", end=" ", flush=True)
    data = rng.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.RandomState(seed + 1).randn(n_queries, dim).astype(np.float32)
    print("done")

    # Cluster
    print("Clustering...", end=" ", flush=True)
    centroids, assignments = simple_kmeans(data, n_clusters, seed=seed)
    centroids = centroids.astype(np.float32)
    assignments = assignments.astype(np.uint32)
    print("done")

    # Ground truth
    print("Computing ground truth...", end=" ", flush=True)
    ground_truth = compute_ground_truth(data, queries, k)
    print("done")

    # Build IVF index
    print("Building IVF index...", end=" ", flush=True)
    cfg = saq.QuantizeConfig()
    cfg.avg_bits = bpd
    cfg.enable_segmentation = True
    cfg.single.quant_type = saq.BaseQuantType.CAQ
    cfg.single.random_rotation = True
    cfg.single.use_fastscan = True
    cfg.single.caq_adj_rd_lmt = 6

    index = saq.IVF(n_vectors, dim, n_clusters, cfg)

    t0 = time.perf_counter()
    index.construct(data, centroids, assignments, num_threads=4)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"done ({build_time:.2f} ms)")

    # Configure search
    search_cfg = saq.SearcherConfig()
    search_cfg.dist_type = saq.DistType.L2Sqr

    # Warmup
    index.search_batch(queries[:10], k, nprobe, search_cfg)

    # Benchmark search
    print("Searching...", end=" ", flush=True)
    t0 = time.perf_counter()
    results = index.search_batch(queries, k, nprobe, search_cfg)
    search_time = (time.perf_counter() - t0) * 1000
    print("done")

    # Compute recall
    recall = compute_recall(results, ground_truth, k)

    # Results
    qps = n_queries / (search_time / 1000)
    print(f"\n=== Results ===")
    print(f"Build Time: {build_time:.2f} ms")
    print(f"Search Time ({n_queries} queries): {search_time:.2f} ms")
    print(f"Queries Per Second: {qps:.2f} QPS")
    print(f"Recall@{k}: {recall * 100:.2f}%")

    return {
        "build_time_ms": build_time,
        "search_time_ms": search_time,
        "qps": qps,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="SAQ Python Benchmark")
    parser.add_argument("--small-only", action="store_true",
                        help="Only run the small-scale benchmark")
    args = parser.parse_args()

    print("=== SAQ Python Bindings Benchmark ===")
    print(f"NumPy version: {np.__version__}")

    configs = [
        {"n_vectors": 10000, "dim": 128, "n_clusters": 100,
         "n_queries": 100, "k": 10, "nprobe": 10, "bpd": 4.0},
    ]

    if not args.small_only:
        configs.extend([
            {"n_vectors": 50000, "dim": 128, "n_clusters": 256,
             "n_queries": 100, "k": 10, "nprobe": 20, "bpd": 4.0},
            {"n_vectors": 100000, "dim": 128, "n_clusters": 512,
             "n_queries": 100, "k": 10, "nprobe": 32, "bpd": 4.0},
        ])

    results = []
    for cfg in configs:
        r = run_benchmark(**cfg)
        results.append(r)

    print("\n\n=== Summary ===")
    print(f"{'Vectors':>10} {'Build(ms)':>12} {'Search(ms)':>12} {'QPS':>12} {'Recall':>10}")
    print("-" * 58)
    for cfg, r in zip(configs, results):
        print(f"{cfg['n_vectors']:>10} {r['build_time_ms']:>12.2f} "
              f"{r['search_time_ms']:>12.2f} {r['qps']:>12.2f} "
              f"{r['recall']*100:>9.2f}%")


if __name__ == "__main__":
    main()
