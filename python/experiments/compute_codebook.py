"""
2x2 Factorial Experiment: Codebook Type x Bit Allocation Cost Model

Conditions:
  A = Uniform codebook + Variance cost model  (BASELINE — current SAQ)
  B = Optimal codebook + Variance cost model   (codebook effect only)
  C = Uniform codebook + Optimal cost model    (allocation effect only)
  D = Optimal codebook + Optimal cost model    (combined)

Measures total reconstruction MSE across all dimensions at each bit budget.

Usage:
    cd python/
    python -m experiments.compute_codebook --data-dir /path/to/dbpedia_100k
"""

import argparse
import time
from pathlib import Path

import numpy as np


# =============================================================================
# I/O
# =============================================================================

def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs file -> (N, D) float32 array."""
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view("float32")


# =============================================================================
# DP-optimal 1D codebook (vectorized)
# =============================================================================

def compute_codebook_dp(values: np.ndarray, max_bits: int = 6):
    """
    DP-optimal contiguous clustering for sorted 1D data.

    Bins data into a weighted histogram, then runs DP with numpy-vectorized
    inner loops. For k clusters and B bins: O(B * k) with vectorized min-search.

    Returns:
        costs: (max_bits+1,) MSE at each bit rate
        codebooks: list of centroid arrays
    """
    sorted_vals = np.sort(values.astype(np.float64))
    n = len(sorted_vals)

    num_bins = min(500, n)
    bin_edges = np.linspace(sorted_vals[0] - 1e-10, sorted_vals[-1] + 1e-10, num_bins + 1)
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

    costs[0] = range_sse(0, B - 1) / n
    codebooks[0] = np.array([range_centroid(0, B - 1)])

    for bits in range(1, max_bits + 1):
        k = 1 << bits
        if k >= B:
            total_sse = bsq.sum() - np.where(bc > 0.5, bs ** 2 / bc, 0).sum()
            centroids = np.where(bc > 0.5, bs / bc, 0)
            costs[bits] = total_sse / n
            codebooks[bits] = centroids
            continue

        INF = 1e30
        prev_dp = np.full(B, INF, dtype=np.float64)
        prev_split = np.zeros(B, dtype=np.int32)

        for i in range(B):
            prev_dp[i] = cost_from_0[i]
            prev_split[i] = 0

        all_splits = [np.zeros(B, dtype=np.int32)]

        for j in range(2, k + 1):
            curr_dp = np.full(B, INF, dtype=np.float64)
            curr_split = np.zeros(B, dtype=np.int32)

            for i in range(j - 1, B):
                m_start = j - 1
                m_end = i + 1
                ms = np.arange(m_start, m_end)

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
            prev_split = curr_split
            all_splits.append(curr_split.copy())

        costs[bits] = prev_dp[B - 1] / n

        centroids = []
        i = B - 1
        for j_idx in range(k - 1, -1, -1):
            m = all_splits[j_idx][i]
            centroids.append(range_centroid(m, i))
            i = m - 1
        centroids.reverse()
        codebooks[bits] = np.array(centroids)

    return costs, codebooks


# =============================================================================
# Quantization MSE functions
# =============================================================================

def uniform_quantize_mse(values: np.ndarray, bits: int, v_max: float) -> float:
    """MSE of uniform scalar quantization with 2^bits levels in [-v_max, v_max]."""
    if bits == 0:
        return np.mean(values ** 2)

    k = 1 << bits
    bin_width = 2.0 * v_max / k
    centroids = np.linspace(-v_max + bin_width / 2, v_max - bin_width / 2, k)

    clipped = np.clip(values, -v_max, v_max)
    bin_idx = np.floor((clipped + v_max) / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, k - 1)
    reconstructed = centroids[bin_idx]

    return float(np.mean((values - reconstructed) ** 2))


def optimal_codebook_mse(values: np.ndarray, codebook: np.ndarray) -> float:
    """MSE when quantizing values using nearest codebook entry."""
    if codebook is None or len(codebook) == 0:
        return float(np.mean(values ** 2))
    cb = np.sort(codebook)
    boundaries = (cb[:-1] + cb[1:]) / 2
    indices = np.searchsorted(boundaries, values)
    reconstructed = cb[indices]
    return float(np.mean((values - reconstructed) ** 2))


# =============================================================================
# Greedy bit allocation
# =============================================================================

BLOCK_SIZE = 32
MAX_QUANT_BIT = 6  # 7-8 unreliable with 500 bins


def greedy_bit_allocation(
    num_blocks: int,
    avg_bits: float,
    total_dims: int,
    cost_fn,  # cost_fn(block_idx, bits) -> float
) -> np.ndarray:
    """
    Greedy per-block bit allocation.

    At each step, allocate 1 bit to the block with the largest cost reduction.
    Each block-bit costs BLOCK_SIZE bits from the total budget.
    """
    total_budget = int(avg_bits * total_dims)
    max_increments = total_budget // BLOCK_SIZE
    allocation = np.zeros(num_blocks, dtype=np.int32)

    for _ in range(max_increments):
        best_gain = -1e-30
        best_block = -1
        for b in range(num_blocks):
            if allocation[b] >= MAX_QUANT_BIT:
                continue
            gain = cost_fn(b, allocation[b]) - cost_fn(b, allocation[b] + 1)
            if gain > best_gain:
                best_gain = gain
                best_block = b
        if best_block < 0 or best_gain <= 0:
            break
        allocation[best_block] += 1

    return allocation


# =============================================================================
# Main experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="2x2 Factorial: Codebook x Allocation")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-vecs", type=int, default=10000)
    parser.add_argument("--num-dims", type=int, default=192)
    parser.add_argument("--dp-samples", type=int, default=5000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()

    # =========================================================================
    # Load data
    # =========================================================================
    print("Loading vectors_pca.fvecs...")
    vectors = read_fvecs(str(data_dir / "vectors_pca.fvecs"))
    N, D = vectors.shape
    print(f"  {N} vectors, {D} dimensions")

    print("Loading variances_pca.fvecs...")
    variances = read_fvecs(str(data_dir / "variances_pca.fvecs")).flatten()
    print(f"  {len(variances)} variance values")

    num_vecs = min(args.num_vecs, N)
    num_dims = min(args.num_dims, D)
    num_blocks = num_dims // BLOCK_SIZE
    usable_dims = num_blocks * BLOCK_SIZE

    print(f"\nExperiment: {num_vecs} vectors, {usable_dims} dims "
          f"({num_blocks} blocks of {BLOCK_SIZE}), max {MAX_QUANT_BIT} bits")

    rng = np.random.default_rng(42)
    if num_vecs < N:
        vec_idx = rng.choice(N, num_vecs, replace=False)
        data = vectors[vec_idx, :usable_dims].copy()
    else:
        data = vectors[:num_vecs, :usable_dims].copy()

    var_flat = variances[:usable_dims]

    # =========================================================================
    # Phase 1: Precompute per-dimension MSE at each bit rate
    # =========================================================================
    print(f"\nPhase 1: Computing codebooks for {usable_dims} dimensions...")
    t_start = time.time()

    # mse_table[codebook_type][dim, bits] = MSE
    uniform_mse_table = np.zeros((usable_dims, MAX_QUANT_BIT + 1))
    optimal_mse_table = np.zeros((usable_dims, MAX_QUANT_BIT + 1))
    dp_samples = min(args.dp_samples, data.shape[0])

    kurtoses = np.zeros(usable_dims)
    skewnesses = np.zeros(usable_dims)

    for d in range(usable_dims):
        col = data[:, d].astype(np.float64)
        v_max = np.max(np.abs(col))
        if v_max < 1e-12:
            continue

        # Distribution stats
        var = np.var(col)
        if var > 1e-15:
            centered = col - np.mean(col)
            kurtoses[d] = np.mean(centered ** 4) / (var ** 2) - 3
            skewnesses[d] = np.mean(centered ** 3) / (var ** 1.5)

        # Subsample for DP
        if len(col) > dp_samples:
            col_sample = col[rng.choice(len(col), dp_samples, replace=False)]
        else:
            col_sample = col

        _, codebooks = compute_codebook_dp(col_sample, MAX_QUANT_BIT)

        for b in range(MAX_QUANT_BIT + 1):
            uniform_mse_table[d, b] = uniform_quantize_mse(col, b, v_max)
            optimal_mse_table[d, b] = optimal_codebook_mse(col, codebooks[b])

        if (d + 1) % 32 == 0:
            elapsed = time.time() - t_start
            rate = (d + 1) / elapsed
            eta = (usable_dims - d - 1) / rate
            print(f"  Dim {d+1}/{usable_dims} ({elapsed:.1f}s, ~{eta:.0f}s left)")

    t_phase1 = time.time() - t_start
    print(f"Phase 1 done: {t_phase1:.1f}s")

    # =========================================================================
    # Phase 2: Build cost functions for bit allocation
    # =========================================================================

    # Block-level variance sums
    block_var = np.array([
        var_flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].sum()
        for i in range(num_blocks)
    ])

    def variance_cost(block_idx, bits):
        """SAQ's current cost model: variance / 2^bits."""
        if bits == 0:
            return block_var[block_idx]
        return block_var[block_idx] / (1 << bits)

    def optimal_cost(block_idx, bits):
        """Cost = actual MSE from optimal codebook at this bit rate."""
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        bits_clamped = min(bits, MAX_QUANT_BIT)
        return optimal_mse_table[start:end, bits_clamped].sum()

    # =========================================================================
    # Phase 3: Run 2x2 factorial experiment
    # =========================================================================
    avg_bits_list = [1.0, 2.0, 4.0]
    conditions = ["A", "B", "C", "D"]
    condition_labels = {
        "A": "Uniform+VarCost",
        "B": "Optimal+VarCost",
        "C": "Uniform+OptCost",
        "D": "Optimal+OptCost",
    }

    # condition -> cost_fn for allocation, mse_table for quantization
    condition_config = {
        "A": (variance_cost, uniform_mse_table),
        "B": (variance_cost, optimal_mse_table),
        "C": (optimal_cost, uniform_mse_table),
        "D": (optimal_cost, optimal_mse_table),
    }

    # results[avg_bits][condition] = total_mse
    results = {}
    # allocations[avg_bits][condition] = allocation array
    allocations = {}

    print(f"\nPhase 2: Running 4 conditions x {len(avg_bits_list)} bit rates...")

    for avg_bits in avg_bits_list:
        results[avg_bits] = {}
        allocations[avg_bits] = {}

        for cond in conditions:
            cost_fn, mse_table = condition_config[cond]

            # Step 1: Allocate bits
            alloc = greedy_bit_allocation(num_blocks, avg_bits, usable_dims, cost_fn)
            allocations[avg_bits][cond] = alloc

            # Step 2: Compute total MSE using allocated bits and condition's codebook
            total_mse = 0.0
            for block_idx in range(num_blocks):
                bits = alloc[block_idx]
                start = block_idx * BLOCK_SIZE
                end = start + BLOCK_SIZE
                bits_clamped = min(bits, MAX_QUANT_BIT)
                total_mse += mse_table[start:end, bits_clamped].sum()

            # Normalize: total MSE per dimension
            results[avg_bits][cond] = total_mse / usable_dims
            allocations[avg_bits][cond] = alloc

    # =========================================================================
    # Report results
    # =========================================================================
    print("\n" + "=" * 80)
    print("TOTAL RECONSTRUCTION MSE (lower is better)")
    print("=" * 80)

    print(f"\n{'avg_bits':>8} | {'A (baseline)':>14} | {'B (codebook)':>14} | "
          f"{'C (allocate)':>14} | {'D (combined)':>14}")
    print("-" * 78)
    for avg_bits in avg_bits_list:
        r = results[avg_bits]
        print(f"{avg_bits:>7.1f}  | {r['A']:>14.8f} | {r['B']:>14.8f} | "
              f"{r['C']:>14.8f} | {r['D']:>14.8f}")

    # Relative improvement
    print("\n" + "=" * 80)
    print("RELATIVE IMPROVEMENT vs BASELINE (lower ratio = better)")
    print("=" * 80)

    print(f"\n{'avg_bits':>8} | {'B/A (codebook)':>14} | {'C/A (alloc)':>14} | "
          f"{'D/A (combined)':>14} | {'Interaction':>14}")
    print("-" * 78)
    for avg_bits in avg_bits_list:
        r = results[avg_bits]
        ba = r["B"] / r["A"] if r["A"] > 1e-15 else 1.0
        ca = r["C"] / r["A"] if r["A"] > 1e-15 else 1.0
        da = r["D"] / r["A"] if r["A"] > 1e-15 else 1.0
        # Interaction: is D/A better or worse than (B/A * C/A)?
        # If independent, D/A ~ B/A * C/A. Ratio > 1 means synergy.
        predicted = ba * ca
        interaction = predicted / da if da > 1e-15 else 1.0
        print(f"{avg_bits:>7.1f}  | {ba:>14.6f} | {ca:>14.6f} | "
              f"{da:>14.6f} | {interaction:>14.6f}")

    print("\n  Interaction > 1 = synergy (combined better than product of individual effects)")
    print("  Interaction < 1 = interference (combined worse than expected)")

    # Per-block allocation table
    print("\n" + "=" * 80)
    print("PER-BLOCK BIT ALLOCATION")
    print("=" * 80)

    for avg_bits in avg_bits_list:
        print(f"\n--- {avg_bits:.1f} bpd ---")
        allocs = allocations[avg_bits]
        print(f"  {'Block':>5} | {'Var(sum)':>10} | "
              f"{'A':>4} | {'B':>4} | {'C':>4} | {'D':>4} | {'Shift?':>6}")
        print("  " + "-" * 55)
        for i in range(num_blocks):
            vsum = var_flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].sum()
            a_b = allocs["A"][i]
            b_b = allocs["B"][i]
            c_b = allocs["C"][i]
            d_b = allocs["D"][i]
            changed = "*" if not (a_b == b_b == c_b == d_b) else ""
            print(f"  {i:>5} | {vsum:>10.4f} | "
                  f"{a_b:>4d} | {b_b:>4d} | {c_b:>4d} | {d_b:>4d} | {changed:>6}")

        # Summary
        for cond in conditions:
            total = allocs[cond].sum() * BLOCK_SIZE
            print(f"  {cond}: {total} total bits "
                  f"(avg {allocs[cond].sum() * BLOCK_SIZE / usable_dims:.2f} bpd)")

    # =========================================================================
    # Per-dimension analysis (kurtosis correlation)
    # =========================================================================
    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS (kurtosis vs optimal codebook advantage)")
    print("=" * 80)

    # Ratio of uniform/optimal MSE at 2 bits per dimension
    ref_bits = 2
    dim_ratios_2b = np.where(
        optimal_mse_table[:, ref_bits] > 1e-15,
        uniform_mse_table[:, ref_bits] / optimal_mse_table[:, ref_bits], 1.0)
    dim_ratios_4b = np.where(
        optimal_mse_table[:, min(4, MAX_QUANT_BIT)] > 1e-15,
        uniform_mse_table[:, min(4, MAX_QUANT_BIT)] / optimal_mse_table[:, min(4, MAX_QUANT_BIT)], 1.0)

    print(f"\n{'Dim':>4} | {'Variance':>10} | {'Kurtosis':>10} | {'Skewness':>8} | "
          f"{'U/O @2b':>8} | {'U/O @4b':>8}")
    print("-" * 68)

    show_dims = sorted(set(
        list(range(min(10, usable_dims))) +
        ([usable_dims // 2 - 1, usable_dims // 2, usable_dims // 2 + 1]
         if usable_dims > 20 else []) +
        list(range(max(0, usable_dims - 3), usable_dims))
    ))

    for d in show_dims:
        print(f"{d:>4} | {np.var(data[:, d]):>10.6f} | {kurtoses[d]:>10.2f} | "
              f"{skewnesses[d]:>8.2f} | {dim_ratios_2b[d]:>8.4f} | {dim_ratios_4b[d]:>8.4f}")

    corr_2b = np.corrcoef(kurtoses, dim_ratios_2b)[0, 1]
    corr_4b = np.corrcoef(kurtoses, dim_ratios_4b)[0, 1]
    print(f"\nCorrelation(kurtosis, U/O ratio @2b) = {corr_2b:.4f}")
    print(f"Correlation(kurtosis, U/O ratio @4b) = {corr_4b:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: {N}x{D} (analyzed {num_vecs}x{usable_dims})")
    print(f"Time: {total_time:.1f}s")

    print(f"\nKey findings:")
    for avg_bits in avg_bits_list:
        r = results[avg_bits]
        ba = (1 - r["B"] / r["A"]) * 100
        ca = (1 - r["C"] / r["A"]) * 100
        da = (1 - r["D"] / r["A"]) * 100
        print(f"  {avg_bits:.0f} bpd: codebook={ba:+.1f}%, allocation={ca:+.1f}%, combined={da:+.1f}%")


if __name__ == "__main__":
    main()
