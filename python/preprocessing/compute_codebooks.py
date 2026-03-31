"""
Precompute optimal codebook centroids per dimension for SAQ.

Outputs two files:
  optimal_costs.fvecs:     D x (max_bits+1) MSE at each bit rate
  optimal_codebooks.fvecs: D x (max_bits+1 * 256) flattened codebook centroids
    Layout: for dim d, bits b: codebooks[d, b*256 : b*256 + 2^b] are the centroids
    Unused entries are 0.

Usage:
    python -m preprocessing.compute_codebooks --data-dir data/datasets/dbpedia_100k
"""

import argparse
import time
from pathlib import Path

import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view("float32")


def write_fvecs(path: str, mat: np.ndarray):
    mat = np.ascontiguousarray(mat, dtype="float32")
    n, d = mat.shape
    out = np.empty((n, d + 1), dtype="float32")
    out[:, 0] = np.frombuffer(
        np.array([d] * n, dtype="int32").tobytes(), dtype="float32"
    )
    out[:, 1:] = mat
    out.tofile(path)


def compute_codebook_dp(values: np.ndarray, max_bits: int = 6,
                        num_bins: int = 500):
    """DP-optimal contiguous clustering. Returns costs and codebook centroids."""
    sorted_vals = np.sort(values.astype(np.float64))
    n = len(sorted_vals)
    num_bins = min(num_bins, n)

    bin_edges = np.linspace(
        sorted_vals[0] - 1e-10, sorted_vals[-1] + 1e-10, num_bins + 1
    )
    bin_indices = np.searchsorted(bin_edges[1:-1], sorted_vals)

    bc = np.zeros(num_bins, dtype=np.float64)
    bs = np.zeros(num_bins, dtype=np.float64)
    bsq = np.zeros(num_bins, dtype=np.float64)
    np.add.at(bc, bin_indices, 1)
    np.add.at(bs, bin_indices, sorted_vals)
    np.add.at(bsq, bin_indices, sorted_vals ** 2)

    mask = bc > 0
    bc, bs, bsq = bc[mask], bs[mask], bsq[mask]
    B = len(bc)

    pc = np.zeros(B + 1, dtype=np.float64)
    ps = np.zeros(B + 1, dtype=np.float64)
    psq = np.zeros(B + 1, dtype=np.float64)
    pc[1:] = np.cumsum(bc)
    ps[1:] = np.cumsum(bs)
    psq[1:] = np.cumsum(bsq)

    def range_sse(a, b):
        c = pc[b + 1] - pc[a]
        if c < 0.5:
            return 0.0
        s = ps[b + 1] - ps[a]
        sq = psq[b + 1] - psq[a]
        return sq - s * s / c

    def range_centroid(a, b):
        c = pc[b + 1] - pc[a]
        if c < 0.5:
            return 0.0
        return (ps[b + 1] - ps[a]) / c

    cost_from_0 = np.zeros(B, dtype=np.float64)
    for i in range(B):
        c = pc[i + 1]
        if c > 0.5:
            cost_from_0[i] = psq[i + 1] - ps[i + 1] ** 2 / c

    costs = np.zeros(max_bits + 1, dtype=np.float64)
    codebooks = [None] * (max_bits + 1)

    # 0 bits
    costs[0] = range_sse(0, B - 1) / n
    codebooks[0] = np.array([range_centroid(0, B - 1)])

    for bits in range(1, max_bits + 1):
        k = 1 << bits
        if k >= B:
            total_sse = bsq.sum() - np.where(bc > 0.5, bs ** 2 / bc, 0).sum()
            costs[bits] = total_sse / n
            codebooks[bits] = np.where(bc > 0.5, bs / bc, 0)[:k]
            continue

        INF = 1e30
        prev_dp = cost_from_0.copy()
        all_splits = [np.zeros(B, dtype=np.int32)]

        for j in range(2, k + 1):
            curr_dp = np.full(B, INF, dtype=np.float64)
            curr_split = np.zeros(B, dtype=np.int32)
            for i in range(j - 1, B):
                ms = np.arange(j - 1, i + 1)
                c_mi = pc[i + 1] - pc[ms]
                s_mi = ps[i + 1] - ps[ms]
                sq_mi = psq[i + 1] - psq[ms]
                sse_mi = np.where(c_mi > 0.5, sq_mi - s_mi ** 2 / c_mi, 0.0)
                prev_costs = np.where(ms > 0, prev_dp[ms - 1], 0.0)
                total = prev_costs + sse_mi
                best_idx = np.argmin(total)
                curr_dp[i] = total[best_idx]
                curr_split[i] = ms[best_idx]
            prev_dp = curr_dp
            all_splits.append(curr_split.copy())

        costs[bits] = prev_dp[B - 1] / n

        # Backtrack to get centroids
        centroids = []
        i = B - 1
        for j_idx in range(k - 1, -1, -1):
            m = all_splits[j_idx][i]
            centroids.append(range_centroid(m, i))
            i = m - 1
        centroids.reverse()
        codebooks[bits] = np.array(centroids)

    return costs, codebooks


def main():
    parser = argparse.ArgumentParser(
        description="Precompute optimal codebooks for SAQ"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--dp-samples", type=int, default=5000)
    parser.add_argument("--num-bins", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    max_bits = args.max_bits
    t0 = time.time()

    print("Loading vectors_pca.fvecs...")
    vectors = read_fvecs(str(data_dir / "vectors_pca.fvecs"))
    N, D = vectors.shape
    print(f"  {N} vectors, {D} dimensions")

    dp_samples = min(args.dp_samples, N)
    rng = np.random.default_rng(42)

    # Normalize each dimension to zero mean for codebook computation
    # (SAQ centers on cluster centroid, so residuals are ~zero mean)
    dim_means = vectors.mean(axis=0)
    dim_stds = vectors.std(axis=0)

    costs_mat = np.zeros((D, max_bits + 1), dtype=np.float32)
    # Codebook layout: D rows, each row has (max_bits+1) * 256 entries
    # For bits b, codebook entries are at [b*256 : b*256 + 2^b]
    cb_cols = (max_bits + 1) * 256
    codebook_mat = np.zeros((D, cb_cols), dtype=np.float32)

    print(f"Computing codebooks for {D} dimensions "
          f"(max_bits={max_bits}, dp_samples={dp_samples})...")

    for d in range(D):
        col = vectors[:, d].astype(np.float64)
        if dim_stds[d] < 1e-12:
            continue

        if len(col) > dp_samples:
            col_sample = col[rng.choice(len(col), dp_samples, replace=False)]
        else:
            col_sample = col

        dim_costs, dim_codebooks = compute_codebook_dp(
            col_sample, max_bits, num_bins=args.num_bins
        )
        costs_mat[d, :] = dim_costs.astype(np.float32)

        for b in range(max_bits + 1):
            cb = dim_codebooks[b]
            if cb is not None:
                k = min(len(cb), 256)
                codebook_mat[d, b * 256: b * 256 + k] = cb[:k].astype(np.float32)

        if (d + 1) % 64 == 0:
            elapsed = time.time() - t0
            rate = (d + 1) / elapsed
            eta = (D - d - 1) / rate
            print(f"  Dim {d+1}/{D} ({elapsed:.1f}s, ~{eta:.0f}s left)")

    # Save
    costs_path = str(data_dir / "optimal_costs.fvecs")
    write_fvecs(costs_path, costs_mat)
    print(f"Saved costs: {costs_mat.shape} to {costs_path}")

    cb_path = str(data_dir / "optimal_codebooks.fvecs")
    write_fvecs(cb_path, codebook_mat)
    print(f"Saved codebooks: {codebook_mat.shape} to {cb_path}")

    print(f"Total time: {time.time() - t0:.1f}s")

    # Sanity: print first dim's codebook at 2 bits
    d = 0
    b = 2
    cb = codebook_mat[d, b*256 : b*256 + (1 << b)]
    print(f"\nDim 0, 2 bits: centroids = {cb}")
    print(f"  (uniform would be evenly spaced in [{vectors[:,0].min():.4f}, {vectors[:,0].max():.4f}])")


if __name__ == "__main__":
    main()
