#pragma once

#include "saq/bit_allocator.h"
#include "saq/defines.h"

#include <Eigen/Dense>

namespace saq {

/// Per-block greedy allocator using empirical per-dim per-bit MSE costs.
///
/// Algorithm: each `dim_padding_size`-dim block starts at 0 bits. Repeatedly find the block
/// whose marginal MSE drop (b -> b+1) is largest and bump it by 1, until the bit
/// budget is exhausted. Final segmentation merges consecutive blocks with identical
/// bit counts. Factor overhead is reserved pessimistically; merging recovers some budget,
/// so the greedy result may use slightly fewer total bits than the DP's exact budget.
class BitAllocatorGreedy : public BitAllocator {
public:
    BitAllocatorGreedy() = default;
    AllocatorKind kind() const override { return AllocatorKind::Greedy; }

    /// `mse_table` is (num_dim_padded, max_bits_per_dim + 1) — mse_table(d, b) is the MSE
    /// for dim d quantized at b bits (sourced from build_codebook_lloyd's costs vector).
    /// Row 0 should be the per-dim variance (the b=0 "no quantization" cost).
    BitAllocationResult AllocateJoint(const Eigen::MatrixXf &mse_table,
                                      const JointAllocationConfig &config) const;
};

}  // namespace saq
