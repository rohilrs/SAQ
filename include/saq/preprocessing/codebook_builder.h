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
    CodebookInit init        = CodebookInit::EqualMassQuantile;
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

/// Build per-dimension Lloyd codebooks for every column of `data` (parallel).
std::vector<CodebookResult> build_all_dims(const FloatRowMat& data,
                                           const LloydOpts& opts);

/// Mean squared reconstruction error of `values` under codebook `cb`
/// (each value mapped to its nearest centroid). Brute-force via cb.nearest.
float codebook_mse(std::span<const float> values, const DimensionCodebook& cb);

}  // namespace saq
