#include "saq/bit_allocator_dp.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace saq {

BitAllocationResult BitAllocatorDP::AllocateJoint(const FloatVec &data_variance,
                                                   const JointAllocationConfig &config) const {
    BitAllocationResult res;

    const size_t num_dim_padded  = config.num_dim_padded;
    const size_t dim_padding     = config.dim_padding_size;
    const size_t max_bits        = config.max_bits_per_dim;
    const size_t tot_bits        = config.total_bits;
    const size_t num_bit_factors = config.num_bit_factors;

    CHECK_EQ(static_cast<size_t>(data_variance.cols()), num_dim_padded);
    CHECK_EQ(num_dim_padded % dim_padding, 0u);

    const size_t num_blocks = num_dim_padded / dim_padding;

    // Back-derive avg_bits to match the existing max_num_segs branch condition.
    // tot_bits == avg_bits * num_dim_padded + num_bit_factors (single global overhead).
    const float avg_bits = (static_cast<float>(tot_bits) - static_cast<float>(num_bit_factors))
                           / static_cast<float>(num_dim_padded);

    const size_t max_num_segs = (avg_bits < 2.0f) ? num_blocks : num_blocks / 2;

    constexpr double valid_lmt = std::numeric_limits<double>::max();

    auto f = std::vector<std::vector<std::vector<std::pair<double, size_t>>>>(
        max_num_segs + 1,
        std::vector<std::vector<std::pair<double, size_t>>>(
            num_blocks + 1,
            std::vector<std::pair<double, size_t>>(tot_bits + 1, {valid_lmt, 0})));

    // Initial best-state: ans_b = tot_bits matches the existing code's initialization,
    // and valid_lmt is a sentinel so any real solution beats it.
    size_t ans_ns = 0;
    size_t ans_i  = num_blocks;
    size_t ans_b  = tot_bits;

    f[0][0][0] = {0.0, 0};

    for (size_t ns = 0; ns <= max_num_segs; ++ns) {
        for (size_t i = 0; i <= num_blocks; ++i) {
            for (size_t used_bits = 0; used_bits <= tot_bits; ++used_bits) {
                if (f[ns][i][used_bits].first < valid_lmt) {
                    if (i == num_blocks) {
                        // Terminal state: check if this beats the current best.
                        // The 1.01 factor matches the existing SaqDataMaker::dynamic_programming.
                        if (f[ns][i][used_bits].first * 1.01 < f[ans_ns][ans_i][ans_b].first) {
                            ans_ns = ns;
                            ans_i  = i;
                            ans_b  = used_bits;
                        }
                        continue;
                    }
                    if (ns == max_num_segs) {
                        continue;
                    }

                    // Try extending with a new segment: j blocks wide, b bits each.
                    double var_sum = 0.0;
                    for (size_t j = 1; (i + j) * dim_padding <= num_dim_padded; ++j) {
                        var_sum += data_variance
                                       .segment(static_cast<Eigen::Index>((i + j - 1) * dim_padding),
                                                static_cast<Eigen::Index>(dim_padding))
                                       .sum();

                        for (size_t b = 1; b <= max_bits; ++b) {
                            const size_t B_new = used_bits + b * j * dim_padding + num_bit_factors;
                            if (B_new > tot_bits) break;
                            const double v = var_sum / static_cast<double>(1u << b);
                            auto &f_to = f[ns + 1][i + j][B_new];
                            if (f_to.first > f[ns][i][used_bits].first + v) {
                                f_to.first  = f[ns][i][used_bits].first + v;
                                f_to.second = (i << 4) + b;
                            }
                        }
                    }

                    // Unquantized-tail branch: assign 0 bits to all remaining dims.
                    // var_sum here reflects all remaining blocks (j iterated to end).
                    const double err0 = var_sum;
                    if (f[ns][i][used_bits].first + err0 <
                        f[ns + 1][num_blocks][used_bits].first) {
                        f[ns + 1][num_blocks][used_bits].first  = f[ns][i][used_bits].first + err0;
                        f[ns + 1][num_blocks][used_bits].second = (i << 4) + 0;
                    }
                }
            }
        }
    }

    // Check that a feasible solution was found.
    if (f[ans_ns][ans_i][ans_b].first == valid_lmt) {
        res.error = "no feasible DP allocation";
        return res;
    }

    // Backtrack to reconstruct the optimal quantization plan.
    BitAllocationResult::QuantPlanT plan;
    {
        size_t ns = ans_ns;
        size_t i  = ans_i;
        size_t B  = ans_b;
        while (i > 0) {
            auto &f_cur = f[ns][i][B];
            const size_t prev_i    = (f_cur.second >> 4);
            const size_t curr_bits = f_cur.second & 0xF;
            const size_t curr_dim_len = (i - prev_i) * dim_padding;
            plan.emplace_back(curr_dim_len, curr_bits);
            ns--;
            i = prev_i;
            if (curr_bits) B -= curr_bits * curr_dim_len + num_bit_factors;
        }
        std::reverse(plan.begin(), plan.end());
    }

    res.quant_plan       = std::move(plan);
    res.total_bits_used  = ans_b;
    res.total_distortion = f[ans_ns][ans_i][ans_b].first;
    return res;
}

}  // namespace saq
