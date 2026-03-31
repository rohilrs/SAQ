"""
Precompute optimal codebook MSE costs per dimension for SAQ's DP bit allocation.

Outputs optimal_costs.fvecs: a D x (max_bits+1) matrix where entry (d, b) is
the MSE of the DP-optimal codebook with 2^b levels for dimension d.

Usage:
    python -m preprocessing.compute_costs --data-dir data/datasets/dbpedia_100k
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs file -> (N, D) float32 array."""
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view("float32")


def write_fvecs(path: str, mat: np.ndarray):
    """Write (N, D) float32 array as .fvecs file."""
    mat = np.ascontiguousarray(mat, dtype="float32")
    n, d = mat.shape
    # Prepend dimension as int32 to each row
    out = np.empty((n, d + 1), dtype="float32")
    out[:, 0] = np.frombuffer(
        np.array([d] * n, dtype="int32").tobytes(), dtype="float32"
    )
    out[:, 1:] = mat
    out.tofile(path)


def compute_codebook_dp(values: np.ndarray, max_bits: int = 8,
                        num_bins: int = 2000):
    """
    DP-optimal contiguous clustering for sorted 1D data.

    Returns:
        costs: (max_bits+1,) MSE at each bit rate
    """
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

    cost_from_0 = np.zeros(B, dtype=np.float64)
    for i in range(B):
        c = pc[i + 1]
        if c > 0.5:
            cost_from_0[i] = psq[i + 1] - ps[i + 1] ** 2 / c

    costs = np.zeros(max_bits + 1, dtype=np.float64)

    # 0 bits: one cluster = mean
    costs[0] = range_sse(0, B - 1) / n

    for bits in range(1, max_bits + 1):
        k = 1 << bits
        if k >= B:
            total_sse = bsq.sum() - np.where(bc > 0.5, bs ** 2 / bc, 0).sum()
            costs[bits] = total_sse / n
            continue

        INF = 1e30
        prev_dp = cost_from_0.copy()

        for j in range(2, k + 1):
            curr_dp = np.full(B, INF, dtype=np.float64)
            for i in range(j - 1, B):
                ms = np.arange(j - 1, i + 1)
                c_mi = pc[i + 1] - pc[ms]
                s_mi = ps[i + 1] - ps[ms]
                sq_mi = psq[i + 1] - psq[ms]
                sse_mi = np.where(c_mi > 0.5, sq_mi - s_mi ** 2 / c_mi, 0.0)
                prev_costs = np.where(ms > 0, prev_dp[ms - 1], 0.0)
                total = prev_costs + sse_mi
                curr_dp[i] = total.min()
            prev_dp = curr_dp

        costs[bits] = prev_dp[B - 1] / n

    return costs


def main():
    parser = argparse.ArgumentParser(
        description="Precompute optimal codebook costs for SAQ DP"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--dp-samples", type=int, default=5000)
    parser.add_argument("--num-bins", type=int, default=2000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()

    print("Loading vectors_pca.fvecs...")
    vectors = read_fvecs(str(data_dir / "vectors_pca.fvecs"))
    N, D = vectors.shape
    print(f"  {N} vectors, {D} dimensions")

    max_bits = args.max_bits
    dp_samples = min(args.dp_samples, N)
    rng = np.random.default_rng(42)

    costs = np.zeros((D, max_bits + 1), dtype=np.float32)

    print(f"Computing optimal costs for {D} dimensions "
          f"(max_bits={max_bits}, dp_samples={dp_samples}, "
          f"num_bins={args.num_bins})...")

    for d in range(D):
        col = vectors[:, d].astype(np.float64)
        v_max = np.max(np.abs(col))
        if v_max < 1e-12:
            continue

        if len(col) > dp_samples:
            col_sample = col[rng.choice(len(col), dp_samples, replace=False)]
        else:
            col_sample = col

        dim_costs = compute_codebook_dp(
            col_sample, max_bits, num_bins=args.num_bins
        )
        costs[d, :] = dim_costs.astype(np.float32)

        if (d + 1) % 64 == 0:
            elapsed = time.time() - t0
            rate = (d + 1) / elapsed
            eta = (D - d - 1) / rate
            print(f"  Dim {d+1}/{D} ({elapsed:.1f}s, ~{eta:.0f}s left)")

    out_path = str(data_dir / "optimal_costs.fvecs")
    write_fvecs(out_path, costs)
    print(f"\nSaved optimal_costs.fvecs: {costs.shape} to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

    # Quick sanity check
    print(f"\nSanity check (first 5 dims, bits 0-4):")
    for d in range(min(5, D)):
        row = costs[d, :5]
        print(f"  dim {d}: {row}")


if __name__ == "__main__":
    main()
