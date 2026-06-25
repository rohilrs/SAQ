#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "saq/codebook_encoder.h"  // DimensionCodebook
#include "saq/defines.h"           // FloatRowMat

namespace saq {

/// Per-dimension result: MSE and codebook at each bit-rate 0..max_bits.
struct CodebookResult {
    // [0..max_bits] reconstruction MSE; values are histogram-approximate —
    // SSE is computed over bin-discretized data, not over the raw float values.
    std::vector<float>             costs;
    std::vector<DimensionCodebook> codebooks;  // [0..max_bits]
};

enum class CodebookInit {
    EqualMassQuantile,
    UniformSpaced,
    KMeansPlusPlus,
    CubeRootDensity
};

struct LloydOpts {
    size_t       max_bits    = 13;
    CodebookInit init        = CodebookInit::KMeansPlusPlus;  // empirically-chosen default (see research-hub/experiments/2026-05-27-codebook-init-sizing.md)
    size_t       restarts    = 1;
    size_t       max_iters   = 50;
    float        tol         = 1e-6f;  // centroid max-move convergence
    uint64_t     seed        = 0;      // KMeansPlusPlus seeding + sampling
    // 0 = use full data; >0 = build the codebook on a deterministic random
    // sample of this many points (seeded by `seed`). The returned centroids
    // are still the deliverable; reported costs are on the sample.
    size_t       sample_size = 0;
};

/// DP-optimal contiguous 1-D clustering (the reference). Valid for max_bits <= 8.
/// Reported costs are MSE over histogram-binned data (bin-discretized), not exact
/// MSE over raw values; accuracy improves with larger num_bins.
CodebookResult build_codebook_dp(std::span<const float> values,
                                 size_t max_bits, size_t num_bins = 500);

/// Fast Lloyd (k-means) construction over a sorted column + prefix sums.
CodebookResult build_codebook_lloyd(std::span<const float> values,
                                    const LloydOpts& opts);

/// Exact (globally optimal) 1-D k-means over the RAW sorted column — no histogram
/// binning. Divide-and-conquer DP optimization (monotone optimal split),
/// O(k n log n); one pass fills all bit-rates 0..max_bits. Exact-optimal contiguous
/// 1-D clustering (Wu 1991 "Optimal Quantization by Matrix Searching"; Grønlund
/// et al. 2017). Costs are exact MSE on the raw values. Intended to replace the
/// histogram-`build_codebook_dp` reference (faster + no num_bins) and, pending an
/// A/B, the Lloyd production codebook.
CodebookResult build_codebook_exact(std::span<const float> values,
                                    size_t max_bits);

/// Build per-dimension Lloyd codebooks for every column of `data` (parallel).
std::vector<CodebookResult> build_all_dims(const FloatRowMat& data,
                                           const LloydOpts& opts);

/// Build per-dimension EXACT codebooks (build_codebook_exact) for every column
/// of `data` (parallel). Drop-in alternative to build_all_dims for the native
/// derivation path — exact-optimal, no num_bins, faster than Lloyd.
std::vector<CodebookResult> build_all_dims_exact(const FloatRowMat& data,
                                                 size_t max_bits);

/// Mean squared reconstruction error of `values` under codebook `cb`
/// (each value mapped to its nearest centroid). Brute-force via cb.nearest.
float codebook_mse(std::span<const float> values, const DimensionCodebook& cb);

/// Returns the smallest sample size that empirically lands kpp-on-sample within
/// ~10% of kpp(full) at the requested bit-rate (see init-sizing experiment).
/// Use when n is much larger than this value to avoid full-data O(n*k) construction.
/// Rule: min(n, max(200_000, 500 * (1<<max_bits))).
constexpr size_t recommended_sample_size(size_t n, size_t max_bits) {
    const size_t k_floor = static_cast<size_t>(500) * (size_t{1} << max_bits);
    const size_t floor   = static_cast<size_t>(200000);
    const size_t lower   = floor > k_floor ? floor : k_floor;
    return n < lower ? n : lower;
}

}  // namespace saq
