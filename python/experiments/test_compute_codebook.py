"""
Unit tests for compute_codebook.py — core quantization and bit allocation functions.

Run with:
    cd python/
    python -m pytest experiments/test_compute_codebook.py -v
"""

import numpy as np
import pytest

from experiments.compute_codebook import (
    BLOCK_SIZE,
    MAX_QUANT_BIT,
    compute_codebook_dp,
    greedy_bit_allocation,
    optimal_codebook_mse,
    uniform_quantize_mse,
)


# =============================================================================
# compute_codebook_dp
# =============================================================================


class TestComputeCodebookDP:
    def test_uniform_4_values_1_bit(self):
        """[1,2,3,4] with 1 bit should split into [1,2] and [3,4] -> centroids ~[1.5, 3.5]."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        costs, codebooks = compute_codebook_dp(values, max_bits=1)

        cb = codebooks[1]
        assert len(cb) == 2
        cb_sorted = np.sort(cb)
        np.testing.assert_allclose(cb_sorted, [1.5, 3.5], atol=0.1)

        # MSE for optimal split: mean of [(1-1.5)^2, (2-1.5)^2, (3-3.5)^2, (4-3.5)^2] = 0.25
        assert costs[1] == pytest.approx(0.25, abs=0.05)

    def test_0_bits_returns_mean(self):
        """0 bits -> single centroid at the mean."""
        values = np.array([2.0, 4.0, 6.0, 8.0])
        costs, codebooks = compute_codebook_dp(values, max_bits=1)

        assert len(codebooks[0]) == 1
        assert codebooks[0][0] == pytest.approx(5.0, abs=0.1)

        # MSE at 0 bits = variance of data
        expected_mse = np.var(values)  # 5.0
        assert costs[0] == pytest.approx(expected_mse, abs=0.1)

    def test_all_identical_values(self):
        """All same values -> MSE = 0 at any bit rate."""
        values = np.full(100, 3.14)
        costs, codebooks = compute_codebook_dp(values, max_bits=3)

        for b in range(4):
            assert costs[b] == pytest.approx(0.0, abs=1e-8)

    def test_gaussian_data_optimal_beats_uniform_mse(self):
        """Optimal DP codebook should have lower MSE than uniform quantization for Gaussian data."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(500).astype(np.float64)
        v_max = np.max(np.abs(values))

        costs, codebooks = compute_codebook_dp(values, max_bits=3)

        for bits in range(1, 4):
            uniform_mse = uniform_quantize_mse(values, bits, v_max)
            # DP-optimal should be <= uniform (it optimizes for contiguous 1D clustering)
            assert costs[bits] <= uniform_mse + 1e-6, (
                f"At {bits} bits, optimal MSE {costs[bits]:.6f} > uniform MSE {uniform_mse:.6f}"
            )

    def test_costs_decrease_with_more_bits(self):
        """More bits should always yield equal or lower MSE."""
        rng = np.random.default_rng(123)
        values = rng.standard_normal(200)
        costs, _ = compute_codebook_dp(values, max_bits=4)

        for b in range(1, 5):
            assert costs[b] <= costs[b - 1] + 1e-10

    def test_many_clusters_cover_all_bins(self):
        """When k >= num_bins, cost should be near-zero (each bin gets its own centroid)."""
        values = np.arange(10, dtype=np.float64)
        costs, _ = compute_codebook_dp(values, max_bits=6)
        # 2^6 = 64 clusters for 10 values -> essentially zero error
        assert costs[6] == pytest.approx(0.0, abs=1e-6)

    def test_bimodal_data_1_bit(self):
        """Bimodal data with clear separation should be perfectly split at 1 bit."""
        values = np.concatenate([np.full(50, -10.0), np.full(50, 10.0)])
        costs, codebooks = compute_codebook_dp(values, max_bits=1)

        cb_sorted = np.sort(codebooks[1])
        np.testing.assert_allclose(cb_sorted, [-10.0, 10.0], atol=0.5)
        assert costs[1] == pytest.approx(0.0, abs=0.1)


# =============================================================================
# uniform_quantize_mse
# =============================================================================


class TestUniformQuantizeMSE:
    def test_data_at_bin_centers_zero_mse(self):
        """If data sits exactly at bin centers, MSE should be 0."""
        v_max = 4.0
        bits = 2  # 4 bins, width = 2.0, centers at -3, -1, 1, 3
        centroids = np.array([-3.0, -1.0, 1.0, 3.0])
        mse = uniform_quantize_mse(centroids, bits, v_max)
        assert mse == pytest.approx(0.0, abs=1e-10)

    def test_0_bits_returns_mean_squared_error(self):
        """0 bits -> reconstruct as 0 (no codebook), so MSE = mean(values^2)."""
        values = np.array([1.0, 2.0, 3.0])
        mse = uniform_quantize_mse(values, 0, v_max=5.0)
        expected = np.mean(values**2)  # (1+4+9)/3 = 14/3
        assert mse == pytest.approx(expected, rel=1e-6)

    def test_symmetric_uniform_data_theoretical_mse(self):
        """For data uniform in [-v_max, v_max], MSE ~ delta^2/12 where delta = 2*v_max/k."""
        v_max = 1.0
        bits = 3  # 8 bins, delta = 0.25
        n = 100000
        rng = np.random.default_rng(99)
        values = rng.uniform(-v_max, v_max, n)

        mse = uniform_quantize_mse(values, bits, v_max)
        delta = 2.0 * v_max / (1 << bits)
        theoretical = delta**2 / 12
        assert mse == pytest.approx(theoretical, rel=0.05)

    def test_mse_decreases_with_bits_from_1(self):
        """For bits >= 1, more bits -> smaller bins -> lower MSE (fixed v_max)."""
        rng = np.random.default_rng(7)
        values = rng.standard_normal(500)
        v_max = np.max(np.abs(values))

        prev_mse = float("inf")
        for bits in range(1, 6):
            mse = uniform_quantize_mse(values, bits, v_max)
            assert mse <= prev_mse + 1e-10
            prev_mse = mse

    def test_clipping_effect(self):
        """Values outside [-v_max, v_max] are clipped, contributing extra error."""
        values = np.array([0.0, 100.0])  # 100 way outside v_max=1
        mse = uniform_quantize_mse(values, 4, v_max=1.0)
        # 100 is clipped to 1.0 then quantized near 1.0; error ~ (100-~0.9375)^2 / 2
        assert mse > 1000  # large error from the outlier


# =============================================================================
# optimal_codebook_mse
# =============================================================================


class TestOptimalCodebookMSE:
    def test_codebook_matches_data_zero_mse(self):
        """If codebook entries exactly match data, MSE = 0."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        codebook = np.array([1.0, 2.0, 3.0, 4.0])
        mse = optimal_codebook_mse(values, codebook)
        assert mse == pytest.approx(0.0, abs=1e-10)

    def test_single_entry_codebook_equals_mean_mse(self):
        """Single centroid at mean -> MSE = variance."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val = np.mean(values)
        codebook = np.array([mean_val])
        mse = optimal_codebook_mse(values, codebook)
        expected = np.mean((values - mean_val) ** 2)
        assert mse == pytest.approx(expected, rel=1e-6)

    def test_nearest_neighbor_assignment(self):
        """Verify nearest-neighbor picks the correct centroid."""
        values = np.array([0.9, 1.1, 2.9, 3.1])
        codebook = np.array([1.0, 3.0])
        mse = optimal_codebook_mse(values, codebook)
        # Errors: (0.1)^2 * 4 items, mean = 0.01
        expected = np.mean([0.01, 0.01, 0.01, 0.01])
        assert mse == pytest.approx(expected, rel=1e-6)

    def test_empty_codebook_returns_mean_squared(self):
        """Empty codebook -> fallback to MSE = mean(values^2)."""
        values = np.array([1.0, 2.0, 3.0])
        mse = optimal_codebook_mse(values, np.array([]))
        expected = np.mean(values**2)
        assert mse == pytest.approx(expected, rel=1e-6)

    def test_none_codebook_returns_mean_squared(self):
        """None codebook -> fallback to MSE = mean(values^2)."""
        values = np.array([1.0, 2.0, 3.0])
        mse = optimal_codebook_mse(values, None)
        expected = np.mean(values**2)
        assert mse == pytest.approx(expected, rel=1e-6)

    def test_unsorted_codebook_still_works(self):
        """Function sorts codebook internally, so unsorted input should work."""
        values = np.array([1.0, 5.0])
        codebook = np.array([5.0, 1.0])  # reversed
        mse = optimal_codebook_mse(values, codebook)
        assert mse == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# greedy_bit_allocation
# =============================================================================


class TestGreedyBitAllocation:
    def test_equal_costs_distribute_evenly(self):
        """If all blocks have equal cost, bits should be distributed roughly evenly."""
        num_blocks = 4
        avg_bits = 2.0
        total_dims = num_blocks * BLOCK_SIZE  # 128

        def equal_cost(block_idx, bits):
            return 10.0 / (1 << bits) if bits > 0 else 10.0

        alloc = greedy_bit_allocation(num_blocks, avg_bits, total_dims, equal_cost)

        # Total budget: 2 * 128 = 256 bits. Each increment costs 32 bits -> 8 increments.
        # With equal cost, each block should get 2 bits.
        total_increments = alloc.sum()
        assert total_increments == 8
        # Even distribution
        np.testing.assert_array_equal(alloc, [2, 2, 2, 2])

    def test_higher_cost_blocks_get_more_bits(self):
        """Block with higher cost should receive more bits first."""
        num_blocks = 2
        avg_bits = 1.0
        total_dims = num_blocks * BLOCK_SIZE  # 64

        costs_per_block = [100.0, 1.0]

        def cost_fn(block_idx, bits):
            return costs_per_block[block_idx] / (1 << bits) if bits > 0 else costs_per_block[block_idx]

        alloc = greedy_bit_allocation(num_blocks, avg_bits, total_dims, cost_fn)

        # Total budget: 1 * 64 = 64 bits -> 2 increments
        # Block 0 has 100x higher cost, so it should get both bits
        assert alloc[0] >= alloc[1]
        assert alloc.sum() == 2

    def test_respects_max_quant_bit_cap(self):
        """No block should exceed MAX_QUANT_BIT even with large budget."""
        num_blocks = 1
        avg_bits = float(MAX_QUANT_BIT + 4)  # way more than cap
        total_dims = BLOCK_SIZE

        def cost_fn(block_idx, bits):
            return 100.0 / (1 << bits) if bits > 0 else 100.0

        alloc = greedy_bit_allocation(num_blocks, avg_bits, total_dims, cost_fn)
        assert alloc[0] <= MAX_QUANT_BIT

    def test_total_bits_within_budget(self):
        """Total allocated bits should not exceed the budget."""
        num_blocks = 6
        avg_bits = 3.0
        total_dims = num_blocks * BLOCK_SIZE

        def cost_fn(block_idx, bits):
            return (block_idx + 1) * 10.0 / (1 << bits) if bits > 0 else (block_idx + 1) * 10.0

        alloc = greedy_bit_allocation(num_blocks, avg_bits, total_dims, cost_fn)
        total_used = alloc.sum() * BLOCK_SIZE
        total_budget = int(avg_bits * total_dims)
        assert total_used <= total_budget

    def test_zero_budget_allocates_nothing(self):
        """With 0 average bits, allocation should be all zeros."""
        num_blocks = 4
        alloc = greedy_bit_allocation(
            num_blocks, 0.0, num_blocks * BLOCK_SIZE,
            lambda b, bits: 10.0 / (1 << bits) if bits > 0 else 10.0,
        )
        np.testing.assert_array_equal(alloc, np.zeros(num_blocks, dtype=np.int32))

    def test_no_gain_stops_early(self):
        """If cost function returns 0 gain, allocation should stop."""
        num_blocks = 2

        def flat_cost(block_idx, bits):
            return 5.0  # no gain from adding bits

        alloc = greedy_bit_allocation(num_blocks, 4.0, num_blocks * BLOCK_SIZE, flat_cost)
        np.testing.assert_array_equal(alloc, [0, 0])


# =============================================================================
# Integration: full 2x2 factorial on synthetic data
# =============================================================================


class TestFactorialIntegration:
    """Run a mini version of the 2x2 factorial on synthetic Gaussian data."""

    @pytest.fixture
    def synthetic_experiment(self):
        """Build MSE tables and run all 4 conditions on small synthetic data."""
        rng = np.random.default_rng(42)
        num_dims = 64  # 2 blocks of 32
        num_vecs = 200
        num_blocks = num_dims // BLOCK_SIZE

        # Generate data: first block has high variance, second has low
        data = np.zeros((num_vecs, num_dims), dtype=np.float64)
        data[:, :32] = rng.standard_normal((num_vecs, 32)) * 5.0
        data[:, 32:] = rng.standard_normal((num_vecs, 32)) * 0.5

        variances = np.var(data, axis=0)
        block_var = np.array([
            variances[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].sum()
            for i in range(num_blocks)
        ])

        max_bits = 4

        uniform_mse_table = np.zeros((num_dims, max_bits + 1))
        optimal_mse_table = np.zeros((num_dims, max_bits + 1))

        for d in range(num_dims):
            col = data[:, d]
            v_max = np.max(np.abs(col))
            if v_max < 1e-12:
                continue
            _, codebooks = compute_codebook_dp(col, max_bits)
            for b in range(max_bits + 1):
                uniform_mse_table[d, b] = uniform_quantize_mse(col, b, v_max)
                optimal_mse_table[d, b] = optimal_codebook_mse(col, codebooks[b])

        def variance_cost(block_idx, bits):
            if bits == 0:
                return block_var[block_idx]
            return block_var[block_idx] / (1 << bits)

        def optimal_cost(block_idx, bits):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            bits_clamped = min(bits, max_bits)
            return optimal_mse_table[start:end, bits_clamped].sum()

        condition_config = {
            "A": (variance_cost, uniform_mse_table),
            "B": (variance_cost, optimal_mse_table),
            "C": (optimal_cost, uniform_mse_table),
            "D": (optimal_cost, optimal_mse_table),
        }

        avg_bits = 2.0
        results = {}
        for cond, (cost_fn, mse_table) in condition_config.items():
            alloc = greedy_bit_allocation(num_blocks, avg_bits, num_dims, cost_fn)
            total_mse = 0.0
            for bi in range(num_blocks):
                bits = min(alloc[bi], max_bits)
                start = bi * BLOCK_SIZE
                end = start + BLOCK_SIZE
                total_mse += mse_table[start:end, bits].sum()
            results[cond] = total_mse / num_dims

        return results

    def test_combined_beats_baseline(self, synthetic_experiment):
        """D (optimal codebook + optimal allocation) should have MSE <= A (baseline)."""
        r = synthetic_experiment
        assert r["D"] <= r["A"] + 1e-10, (
            f"Combined D={r['D']:.6f} should be <= baseline A={r['A']:.6f}"
        )

    def test_optimal_codebook_beats_uniform(self, synthetic_experiment):
        """B (optimal codebook) should have MSE <= A (uniform codebook), same allocation."""
        r = synthetic_experiment
        assert r["B"] <= r["A"] + 1e-10, (
            f"Optimal codebook B={r['B']:.6f} should be <= uniform A={r['A']:.6f}"
        )

    def test_all_conditions_positive_mse(self, synthetic_experiment):
        """All conditions should produce non-negative MSE."""
        for cond, mse in synthetic_experiment.items():
            assert mse >= 0, f"Condition {cond} has negative MSE: {mse}"

    def test_d_is_best_or_tied(self, synthetic_experiment):
        """D should be the best (or tied) across all conditions."""
        r = synthetic_experiment
        assert r["D"] <= min(r["A"], r["B"], r["C"]) + 1e-10
