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
    std::vector<float>             costs;      // [0..max_bits] reconstruction MSE
    std::vector<DimensionCodebook> codebooks;  // [0..max_bits]
};

enum class CodebookInit { EqualMassQuantile, UniformSpaced, KMeansPlusPlus };

struct LloydOpts {
    size_t       max_bits  = 13;
    CodebookInit init      = CodebookInit::EqualMassQuantile;
    size_t       restarts  = 1;
    size_t       max_iters = 50;
    float        tol        = 1e-6f;  // centroid max-move convergence
    uint64_t     seed       = 0;      // only used for KMeansPlusPlus
};

/// DP-optimal contiguous 1-D clustering (the reference). Valid for max_bits <= 8.
CodebookResult build_codebook_dp(std::span<const float> values,
                                 size_t max_bits, size_t num_bins = 500);

/// Fast Lloyd (k-means) construction over a sorted column + prefix sums.
CodebookResult build_codebook_lloyd(std::span<const float> values,
                                    const LloydOpts& opts);

/// Build per-dimension Lloyd codebooks for every column of `data` (parallel).
std::vector<CodebookResult> build_all_dims(const FloatRowMat& data,
                                           const LloydOpts& opts);

}  // namespace saq
