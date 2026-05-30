#include "saq/bit_allocator_greedy.h"

#include <glog/logging.h>

#include <algorithm>
#include <vector>

namespace saq {

BitAllocationResult BitAllocatorGreedy::AllocateJoint(const Eigen::MatrixXf &mse_table,
                                                     const JointAllocationConfig &config) const {
    BitAllocationResult res;
    const size_t num_dim_padded  = config.num_dim_padded;
    const size_t dim_padding     = config.dim_padding_size;
    const size_t max_bits        = config.max_bits_per_dim;
    const size_t tot_bits        = config.total_bits;
    const size_t num_bit_factors = config.num_bit_factors;

    CHECK_EQ(static_cast<size_t>(mse_table.rows()), num_dim_padded);
    CHECK_GE(static_cast<size_t>(mse_table.cols()), max_bits + 1);
    CHECK_EQ(num_dim_padded % dim_padding, 0u);

    const size_t num_blocks = num_dim_padded / dim_padding;

    // Reserve factor overhead pessimistically (one factor per block); merging recovers some.
    // This means the greedy result may use slightly fewer total bits than the DP's exact budget.
    const size_t budget_after_factors = (tot_bits > num_blocks * num_bit_factors)
        ? tot_bits - num_blocks * num_bit_factors : 0;
    const size_t block_bit_units = budget_after_factors / dim_padding;

    // Precompute per-block MSE at each bit level: block_mse[i][b].
    std::vector<std::vector<double>> block_mse(num_blocks,
                                                std::vector<double>(max_bits + 1, 0.0));
    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t b = 0; b <= max_bits; ++b) {
            double s = 0.0;
            for (size_t k = 0; k < dim_padding; ++k) {
                s += static_cast<double>(mse_table(
                    static_cast<Eigen::Index>(i * dim_padding + k),
                    static_cast<Eigen::Index>(b)));
            }
            block_mse[i][b] = s;
        }
    }

    // Greedy loop.
    std::vector<size_t> bits(num_blocks, 0);
    for (size_t step = 0; step < block_bit_units; ++step) {
        ssize_t best_i = -1;
        double  best_drop = -1.0;
        for (size_t i = 0; i < num_blocks; ++i) {
            if (bits[i] >= max_bits) continue;
            double drop = block_mse[i][bits[i]] - block_mse[i][bits[i] + 1];
            if (drop > best_drop) { best_drop = drop; best_i = static_cast<ssize_t>(i); }
        }
        if (best_i < 0) break;  // all blocks capped
        bits[static_cast<size_t>(best_i)]++;
    }

    // Merge consecutive equal-bit blocks into segments.
    BitAllocationResult::QuantPlanT plan;
    double total_dist = 0.0;
    size_t total_bits_used = 0;
    for (size_t i = 0; i < num_blocks;) {
        const size_t b = bits[i];
        size_t j = i + 1;
        while (j < num_blocks && bits[j] == b) ++j;
        const size_t dim_len = (j - i) * dim_padding;
        plan.emplace_back(dim_len, b);
        for (size_t k = i; k < j; ++k) total_dist += block_mse[k][b];
        total_bits_used += b * dim_len + (b > 0 ? num_bit_factors : 0);
        i = j;
    }

    res.quant_plan       = std::move(plan);
    res.total_bits_used  = total_bits_used;
    res.total_distortion = total_dist;
    return res;
}

}  // namespace saq
